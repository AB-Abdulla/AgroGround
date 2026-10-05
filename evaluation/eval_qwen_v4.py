"""
Main inference script on the human-verified benchmark (1,480 images).

Runs a fine-tuned or zero-shot Qwen3-VL model under one of the two protocols and
writes a prediction file in the standard format (coco_id, granularity, dataset,
gt_boxes, pred_boxes_norm, pred_labels, image_width, image_height, raw_output):
  --protocol given        known-target grounding: the prompt names the disease and
                          asks for all affected regions (lesion-level) or the
                          affected leaf or plant (object-level); on healthy images
                          the prompt names a crop-matched disease and the correct
                          output is an empty list
  --protocol recognition  candidate protocol: the prompt lists the four candidates
                          from recognition_eval_manifest_v4.jsonl and asks the model
                          to identify the one present and localize it

Boxes are parsed from the model's JSON output in its native 0-1000 space. No pixel
cap is applied at evaluation. Resumable; existing predictions are skipped.

Usage: python eval_qwen_v4.py --adapter <LoRA dir or none> [--model <merged model dir>]
       --protocol given|recognition --gpu <id> --out <prediction file>
"""
import argparse, os
p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--protocol", choices=["given", "recognition"], required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--out", required=True); p.add_argument("--mem_fraction", type=float, default=None)
p.add_argument("--only_ids", default=None, help="json list of coco_ids to (re)run; other rows kept from existing --out")
p.add_argument("--model", default=None, help="full HF model dir; overrides base (use with --adapter none)")
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json, re, time, torch
if args.mem_fraction: torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0); print(f"[cap] GPU memory fraction {args.mem_fraction}", flush=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

GT_V2 = "./work/human_verified_ground_truth_v4.jsonl"
MAN_V2 = "./work/recognition_eval_manifest_v4.jsonl"
BASE = "./weights/qwen3-vl-2b"

def fix_path(pth):
    if pth and '/home/jovyan' in pth:
        pth = pth.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                          './datasets/cddm/images')
    return pth

def given_prompt(g):
    disease = ((g.get('abstention_disease') if g['granularity']=='healthy' else g.get('disease_raw')) or 'disease').strip().lower()
    if g['granularity'] == 'coarse':
        return (f"Locate the leaf or plant affected by {disease} in this image. "
                f"Output the bounding box in JSON format. If not present, output an empty list.")
    return (f"Locate all regions affected by {disease} in this image. "
            f"Output bounding boxes in JSON format. If no such regions are present, output an empty list.")

def parse_pred(text):
    boxes, labels = [], []
    try:
        for item in json.loads(text):
            if isinstance(item, dict) and 'bbox_2d' in item and len(item['bbox_2d']) == 4:
                boxes.append([float(x) for x in item['bbox_2d']])
                labels.append(str(item.get('label', '')))
        return boxes, labels
    except: pass
    for m in re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text):
        boxes.append([float(x) for x in m]); labels.append('')
    return boxes, labels

if args.protocol == "given":
    items = [json.loads(l) for l in open(GT_V2)]
    items = [g for g in items if g['granularity'] in ('fine','coarse','healthy')]
    get_img = lambda r: fix_path(r['orig_path'])
    get_txt = lambda r: given_prompt(r)
else:
    items = [json.loads(l) for l in open(MAN_V2)]
    get_img = lambda r: fix_path(r['orig_path'])
    get_txt = lambda r: r['prompt']

print(f"[{args.out}] loading base + adapter {args.adapter} on GPU {args.gpu}")
model = Qwen3VLForConditionalGeneration.from_pretrained(args.model or BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
if args.adapter.lower()!="none": model = PeftModel.from_pretrained(model, args.adapter)
else: print(f"[{args.out}] ZERO-SHOT base model, no adapter", flush=True)
model.eval()
processor = AutoProcessor.from_pretrained(args.model or BASE)
print(f"[{args.out}] running {len(items)} images, protocol={args.protocol}")

PARTIAL = f"./work/{args.out}.partial"
results = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []
done_ids = {r["coco_id"] for r in results}
if args.only_ids:
    _ids=set(json.load(open(args.only_ids))); _prev=json.load(open(f"./work/{args.out}"))
    results=[r for r in _prev if r["coco_id"] not in _ids]; done_ids={r["coco_id"] for r in results}; items=[it for it in items if it["coco_id"] in _ids]
    print(f"[{args.out}] re-running {len(items)} ids, keeping {len(results)} rows", flush=True)
if done_ids: print(f"[{args.out}] resuming: {len(done_ids)} already done", flush=True)
t0 = time.time()
for i, r in enumerate(items):
    if r["coco_id"] in done_ids: continue
    if len(results) % 100 == 0 and len(results) > 0 and i % 100 == 0:
        json.dump(results, open(PARTIAL, "w"))
    if i % 100 == 0 and i > 0:
        el = time.time() - t0
        print(f"  {i}/{len(items)} ({el/60:.1f} min, ETA {el/i*(len(items)-i)/60:.0f} min)", flush=True)
    try:
        messages = [{"role":"user","content":[{"type":"image","image":get_img(r)},
                                              {"type":"text","text":get_txt(r)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        out = processor.batch_decode([o[len(i_):] for i_, o in zip(inputs.input_ids, gen)],
                                     skip_special_tokens=True)[0]
        pred_boxes, pred_labels = parse_pred(out)
    except Exception as e:
        out = f"ERROR: {e}"; pred_boxes, pred_labels = [], []
    row = {'coco_id': r['coco_id'], 'granularity': r['granularity'], 'dataset': r['dataset'],
           'gt_boxes': r['gt_boxes'], 'pred_boxes_norm': pred_boxes, 'pred_labels': pred_labels,
           'image_width': r['image_width'], 'image_height': r['image_height'],
           'raw_output': out[:250]}
    if args.protocol == "recognition":
        row['candidates'] = r['candidates']; row['correct_answer'] = r['correct_answer']
    results.append(row)

with open(f"./work/{args.out}", "w") as f:
    json.dump(results, f)
if os.path.exists(PARTIAL): os.remove(PARTIAL)
print(f"[{args.out}] DONE in {(time.time()-t0)/60:.1f} min -> ./work/{args.out}")
