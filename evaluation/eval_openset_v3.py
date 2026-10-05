"""
Open-set grounded recognition on the benchmark (no candidate list), first prompt formulation.

The model is asked to name the disease or pest itself and to box the evidence, or to
return an empty list on a healthy image. This is the first of the two prompt formulations
reported in the paper's open-set appendix; eval_openset_v4.py is the second (its prompt
gives example names). Output is a prediction file in the standard format, scored with the
lenient and strict name-matching rules described in the appendix. Resumable.

Usage: python evaluation/eval_openset_v3.py --adapter <LoRA dir or none> [--model <merged model dir>]
       --gpu <id> --out <prediction file>
"""
import argparse, json, os, re, time
p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True, help="LoRA dir or 'none'"); p.add_argument("--model", default=None, help="full HF model dir (overrides base)")
p.add_argument("--gpu", default="0"); p.add_argument("--out", required=True)
args = p.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
BASE = "./weights/qwen3-vl-2b"; GT = "./work/human_verified_ground_truth_v4.jsonl"
PROMPT = ("Identify the disease, pest, or condition affecting this plant and locate all affected regions. "
          "Output bounding boxes in JSON format with the identified name as the label. If the plant is healthy, output an empty list.")
def fix_path(pth): return pth.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if pth and '/home/jovyan' in pth else pth
def parse(out):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip()).strip()
    try: data = json.loads(s)
    except Exception: return [], [], False
    if not isinstance(data, list): return [], [], False
    boxes, labels = [], []
    for it in data:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4: boxes.append([float(x) for x in b]); labels.append(str(it.get("label", "")))
    return boxes, labels, True
MODEL_DIR = args.model or BASE
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0")
if args.adapter.lower() != "none":
    from peft import PeftModel; model = PeftModel.from_pretrained(model, args.adapter)
model.eval(); processor = AutoProcessor.from_pretrained(MODEL_DIR)
items = [json.loads(l) for l in open(GT)]
PARTIAL = f"./work/{args.out}.partial"
results = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []; done = {r["coco_id"] for r in results}
print(f"[{args.out}] open-set, {len(items)} items, model={MODEL_DIR}, adapter={args.adapter}, resuming {len(done)}", flush=True); t0 = time.time()
for i, g in enumerate(items):
    if g["coco_id"] in done: continue
    if i % 100 == 0 and results: json.dump(results, open(PARTIAL, "w"))
    if i % 100 == 0 and i > 0: print(f"  {i}/{len(items)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    msgs = [{"role":"user","content":[{"type":"image","image":fix_path(g["orig_path"])},{"type":"text","text":PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); ii, vi = process_vision_info(msgs)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda:0")
    try:
        with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        out = processor.batch_decode([gen[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]
    except Exception as e: out = f"ERROR: {e}"
    boxes, labels, ok = parse(out)
    results.append({"coco_id": g["coco_id"], "granularity": g["granularity"], "dataset": g["dataset"], "gt_boxes": g["gt_boxes"],
                    "image_width": g["image_width"], "image_height": g["image_height"], "gold_label": g.get("disease_raw"),
                    "pred_boxes_norm": boxes, "pred_labels": labels, "format_ok": ok, "raw_output": out[:300]})
json.dump(results, open(f"./work/{args.out}", "w"))
if os.path.exists(PARTIAL): os.remove(PARTIAL)
print(f"[{args.out}] DONE in {(time.time()-t0)/60:.1f} min -> ./work/{args.out}", flush=True)
