"""
Control experiments on the benchmark positives (975 images).

Three modes, all producing prediction files in the standard format:
  mode blank: candidate protocol with the image replaced by a uniform grey image of the same size (tests for a text-only prior)
  mode shuffle: candidate protocol with the image cut into a 4x4 grid of tiles and the tiles randomly permuted (tests reliance on local texture versus global structure)
  mode hardneg: known-target query naming a plausible disease of the same crop that is not present on the image, the correct output is an empty box list (tests named-target abstention on diseased images)

The hard-negative disease choice and the tile permutation are drawn from one
random stream seeded with 23. Resumable.

Usage: python eval_control_v4.py --adapter <LoRA dir or none> [--model <merged model dir>]
       mode blank|shuffle|hardneg --gpu <id> --out <prediction file>
"""
import argparse, json, os, re, time, random

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True, help="LoRA dir or 'none'")
p.add_argument("--model", default=None, help="full HF model dir (overrides base)")
p.add_argument("--mode", choices=["blank", "shuffle", "hardneg"], required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--out", required=True)
p.add_argument("--mem_fraction", type=float, default=None)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
if args.mem_fraction:
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

BASE = "./weights/qwen3-vl-2b"
GT   = "./work/human_verified_ground_truth_v4.jsonl"
MAN  = "./work/recognition_eval_manifest_v4.jsonl"

def fixp(p):
    return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                     './datasets/cddm/images') if p and '/home/jovyan' in p else p
def norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()
def too_similar(a, b):
    na, nb = norm(a), norm(b)
    return na == nb or na in nb or nb in na

random.seed(23)
gold = {json.loads(l)["coco_id"]: json.loads(l) for l in open(GT)}
man  = {json.loads(l)["coco_id"]: json.loads(l) for l in open(MAN)}
pos  = [i for i, g in gold.items() if g["granularity"] != "healthy"]

pool = {}
for i in pos:
    c = (gold[i].get("crop") or "").strip().lower()
    d = norm(gold[i].get("disease_raw"))
    if c and d:
        pool.setdefault(c, set()).add(d)
alld = sorted({norm(gold[i].get("disease_raw")) for i in pos if gold[i].get("disease_raw")})

def hard_negative(i):
    g = gold[i]
    c = (g.get("crop") or "").strip().lower()
    truth = norm(g.get("disease_raw"))
    cands = [d for d in sorted(pool.get(c, set())) if not too_similar(d, truth)]
    if not cands:
        cands = [d for d in alld if not too_similar(d, truth)]
    return random.choice(cands) if cands else None

def transform(img, mode):
    if mode == "blank":
        return Image.new("RGB", img.size, (128, 128, 128))
    if mode == "shuffle":
        W, H = img.size
        w, h = W // 4, H // 4
        tiles = [img.crop((c * w, r * h, (c + 1) * w, (r + 1) * h)) for r in range(4) for c in range(4)]
        random.shuffle(tiles)
        out = Image.new("RGB", (w * 4, h * 4))
        for k, t in enumerate(tiles):
            out.paste(t, ((k % 4) * w, (k // 4) * h))
        return out
    return img

MODEL_DIR = args.model or BASE
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0")
if args.adapter.lower() != "none":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_DIR)

PARTIAL = f"./work/{args.out}.partial"
results = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []
done = {r["coco_id"] for r in results}
print(f"[{args.out}] mode={args.mode}, {len(pos)} positives, model={MODEL_DIR}, adapter={args.adapter}, resuming {len(done)}", flush=True)
t0 = time.time()

for n, i in enumerate(pos):
    if i in done:
        continue
    if n % 100 == 0 and results:
        json.dump(results, open(PARTIAL, "w"))
    if n % 100 == 0 and n > 0:
        print(f"  {n}/{len(pos)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    g = gold[i]
    img = Image.open(fixp(g["orig_path"])).convert("RGB")
    rec = {"coco_id": i, "granularity": g["granularity"], "dataset": g["dataset"], "gt_boxes": g["gt_boxes"],
           "image_width": g["image_width"], "image_height": g["image_height"]}
    if args.mode == "hardneg":
        d = hard_negative(i)
        if not d:
            continue
        prompt = (f"Locate the {d} in this image. Output the bounding boxes of the affected regions in JSON format. "
                  f"If it is not present, output an empty list.")
        rec.update({"queried_disease": d, "true_disease": g.get("disease_raw")})
        use = img
    else:
        m = man[i]
        cands = ", ".join(norm(c) for c in m["candidates"])
        prompt = (f"This plant may be affected by one of the following: {cands}. "
                  f"Identify which one is present in the image and locate all affected regions. "
                  f"Output bounding boxes in JSON format with the identified name as the label.")
        rec.update({"candidates": m["candidates"], "correct_answer": m["correct_answer"]})
        use = transform(img, args.mode)
    msgs = [{"role": "user", "content": [{"type": "image", "image": use}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda:0")
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        out = processor.batch_decode([gen[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]
    except Exception as e:
        out = f"ERROR: {e}"
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip()).strip()
    boxes, labels = [], []
    try:
        data = json.loads(s)
        if isinstance(data, list):
            for it in data:
                b = it.get("bbox_2d") if isinstance(it, dict) else None
                if isinstance(b, list) and len(b) == 4:
                    boxes.append([float(x) for x in b])
                    labels.append(str(it.get("label", "")))
    except Exception:
        pass
    rec.update({"pred_boxes_norm": boxes, "pred_labels": labels, "raw_output": out[:300]})
    results.append(rec)

json.dump(results, open(f"./work/{args.out}", "w"))
if os.path.exists(PARTIAL):
    os.remove(PARTIAL)
print(f"[{args.out}] DONE in {(time.time()-t0)/60:.1f} min -> ./work/{args.out}", flush=True)
