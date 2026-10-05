"""
Reasoning-grounding baselines on the benchmark: VisionReasoner-7B and Seg-Zero-7B.

Both models are run with their published prompt templates and output parsers,
at their native 840x840 input size, with boxes scaled back to original pixels,
and the results are written in the standard prediction format. VisionReasoner
emits multiple boxes (--mode multi); Seg-Zero emits one (--mode single). Under the
candidate protocol, which these models cannot attempt natively, the candidate
first mentioned in the model's reasoning text is taken as its answer.

Usage: python eval_reasoner_v4.py --model_path <model dir> --mode multi|single
       --protocol given|recognition --gpu <id> --out <prediction file>
Optional: --mem_fraction <0-1>, --only_ids <json list>.
"""
import argparse, json, os, re, time
p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True); p.add_argument("--mode", choices=["multi","single"], required=True)
p.add_argument("--protocol", choices=["given","recognition"], required=True); p.add_argument("--gpu", default="0"); p.add_argument("--out", required=True); p.add_argument("--mem_fraction", type=float, default=None); p.add_argument("--only_ids", default=None)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
if args.mem_fraction: torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0); print(f"[cap] GPU memory fraction {args.mem_fraction}", flush=True)
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

GT_V3 = "./work/human_verified_ground_truth_v4.jsonl"; MAN_V3 = "./work/recognition_eval_manifest_v4.jsonl"
RESIZE = 840
def fix_path(pth): return pth.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if pth and '/home/jovyan' in pth else pth
def norm(s): return re.sub(r"\s+"," ",str(s or "")).strip().lower()

TEMPLATE_MULTI = ('Please find "{Question}" with bboxs and points.'
    'Compare the difference between object(s) and find the most closely matched object(s).'
    'Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'
    'Output the bbox(es) and point(s) inside the interested object(s) in JSON format.'
    'i.e., <think> thinking process here </think><answer>{Answer}</answer>')
ANSWER_MULTI = '[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}, {"bbox_2d": [225,296,706,786], "point_2d": [302,410]}]'
TEMPLATE_SINGLE = ("Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think><answer>{Answer}</answer>")
ANSWER_SINGLE = "{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}"

def question_for(item):
    if args.protocol == "given":
        d = norm((item.get("abstention_disease") if item["granularity"]=="healthy" else item.get("disease_raw")) or "disease")
        return f"the leaf or plant affected by {d}" if item["granularity"] == "coarse" else f"regions affected by {d}"
    cands = ", ".join(norm(c) for c in item["candidates"])
    return f"the regions affected by the disease present in this plant, which is one of: {cands}"

def parse(output, xf, yf):
    boxes = []
    if args.mode == "multi":
        m = re.search(r'<answer>\s*(.*?)\s*</answer>', output, re.DOTALL)
        try:
            data = json.loads(m.group(1)) if m else []
            for it in data:
                b = it.get("bbox_2d")
                if isinstance(b, list) and len(b) == 4: boxes.append([b[0]*xf, b[1]*yf, b[2]*xf, b[3]*yf])
        except Exception: pass
    else:
        m = re.search(r'{[^}]+}', output)
        if m:
            try:
                data = json.loads(m.group(0).replace("'", '"'))
                k = next((k for k in data if "bbox" in k.lower()), None)
                if k and len(data[k]) == 4: b = data[k]; boxes.append([b[0]*xf, b[1]*yf, b[2]*xf, b[3]*yf])
            except Exception: pass
    return boxes

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0").eval()
processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
items = [json.loads(l) for l in open(MAN_V3 if args.protocol == "recognition" else GT_V3)]
PARTIAL = f"./work/{args.out}.partial"
results = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []; done = {r["coco_id"] for r in results}
if args.only_ids:
    _ids=set(json.load(open(args.only_ids))); _prev=json.load(open(f"./work/{args.out}")); results=[r for r in _prev if r["coco_id"] not in _ids]; done={r["coco_id"] for r in results}; items=[it for it in items if it["coco_id"] in _ids]; print(f"[{args.out}] re-running {len(items)} ids, keeping {len(results)}", flush=True)
print(f"[{args.out}] {args.mode} model, protocol={args.protocol}, {len(items)} items, resuming {len(done)}", flush=True)
t0 = time.time()
for i, it in enumerate(items):
    if it["coco_id"] in done: continue
    if i % 50 == 0 and results: json.dump(results, open(PARTIAL, "w"))
    if i % 100 == 0 and i > 0: print(f"  {i}/{len(items)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    img = Image.open(fix_path(it["orig_path"])).convert("RGB"); W, H = img.size
    tmpl, ans = (TEMPLATE_MULTI, ANSWER_MULTI) if args.mode == "multi" else (TEMPLATE_SINGLE, ANSWER_SINGLE)
    msgs = [{"role":"user","content":[{"type":"image","image":img.resize((RESIZE,RESIZE), Image.BILINEAR)},{"type":"text","text":tmpl.format(Question=question_for(it), Answer=ans)}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda:0")
    try:
        with torch.no_grad(): gen = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        out = processor.batch_decode([gen[0][len(inputs.input_ids[0]):]], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except Exception as e: out = f"ERROR: {e}"
    boxes = parse(out, W/RESIZE, H/RESIZE)
    rec = {"coco_id": it["coco_id"], "granularity": it["granularity"], "dataset": it["dataset"], "gt_boxes": it["gt_boxes"],
           "image_width": W, "image_height": H, "pred_boxes_pixel": boxes, "raw_output": out[:500]}
    if args.protocol == "recognition":
        rec["candidates"] = it["candidates"]; rec["correct_answer"] = it["correct_answer"]
        low = norm(out); hits = [c for c in it["candidates"] if norm(c) in low]
        rec["pred_label"] = min(hits, key=lambda c: low.find(norm(c))) if hits else None   # first-mentioned candidate
    results.append(rec)
json.dump(results, open(f"./work/{args.out}", "w"))
if os.path.exists(PARTIAL): os.remove(PARTIAL)
print(f"[{args.out}] DONE in {(time.time()-t0)/60:.1f} min -> ./work/{args.out}", flush=True)
