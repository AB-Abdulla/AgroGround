"""
GroundingDINO recognition baseline on the benchmark (detector as classifier).

GroundingDINO cannot answer "which disease?" natively. The adaptation used in the
paper runs it once per candidate of the recognition manifest; the candidate whose
best detection has the highest confidence is its answer, and that candidate's
boxes are its grounding. If no candidate yields a box above threshold, the output
is "none" (abstention). Same thresholds as the grounding baseline (box 0.25, text
0.20); the image is loaded once and predicted four times.

Input: recognition_eval_manifest_v4.jsonl in the work directory.
Output: gdino_recog_v4.json in the standard prediction format.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json, time
import torch

WEIGHTS_DIR   = "./weights"
GDINO_CONFIG  = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
MANIFEST = "./work/recognition_eval_manifest_v4.jsonl"
OUTPUT   = "./work/gdino_recog_v4.json"
BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.20

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

print("[Model] Loading GroundingDINO on GPU 2...")
from groundingdino.util.inference import load_model, load_image, predict
model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
print("[Model] Loaded")

recs = [json.loads(l) for l in open(MANIFEST)]
print(f"Running detector-as-classifier on {len(recs)} images x {len(recs[0]['candidates'])} candidates...")

results = []
t0 = time.time()
for i, r in enumerate(recs):
    if i % 100 == 0 and i > 0:
        el = time.time() - t0
        print(f"  {i}/{len(recs)} ({el/60:.1f} min, ETA {el/i*(len(recs)-i)/60:.0f} min)")
    img_path = fix_path(r['orig_path'])
    W, H = r['image_width'], r['image_height']
    best = {"label": None, "score": -1.0, "boxes": []}
    cand_scores = {}
    try:
        image_source, image = load_image(img_path)
        Hs, Ws = image_source.shape[:2]
        for cand in r['candidates']:
            boxes, logits, phrases = predict(
                model=model, image=image, caption=cand.lower().strip(),
                box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device="cuda")
            if len(logits) == 0:
                cand_scores[cand] = 0.0
                continue
            score = float(logits.max())
            cand_scores[cand] = score
            if score > best["score"]:
                px = []
                for b in boxes:
                    cx, cy, bw, bh = b.tolist()
                    px.append([round((cx-bw/2)*Ws), round((cy-bh/2)*Hs),
                               round((cx+bw/2)*Ws), round((cy+bh/2)*Hs)])
                best = {"label": cand, "score": score, "boxes": px}
    except Exception as e:
        print(f"  error on {img_path}: {e}")

    results.append({
        'coco_id': r['coco_id'], 'granularity': r['granularity'], 'dataset': r['dataset'],
        'candidates': r['candidates'], 'correct_answer': r['correct_answer'],
        'gt_boxes': r['gt_boxes'],
        'pred_label': best["label"],            # None = abstained (no candidate fired)
        'pred_score': best["score"],
        'pred_boxes_pixel': best["boxes"],
        'candidate_scores': cand_scores,
        'image_width': W, 'image_height': H,
    })

with open(OUTPUT, "w") as f:
    json.dump(results, f)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved {len(results)}.")

# Quick recognition tally
def norm(s): return (s or "").strip().lower()
pos = [x for x in results if x['granularity'] != 'healthy']
correct = sum(1 for x in pos if x['pred_label'] and norm(x['pred_label']) == norm(x['correct_answer']))
abstain_pos = sum(1 for x in pos if x['pred_label'] is None)
healthy = [x for x in results if x['granularity'] == 'healthy']
healthy_abstain = sum(1 for x in healthy if x['pred_label'] is None)
print(f"\nQUICK TALLY — GDino recognition (positive n={len(pos)}, chance=25%):")
print(f"  correct: {correct} ({correct/len(pos)*100:.1f}%) | abstained on positives: {abstain_pos}")
print(f"  healthy abstention: {healthy_abstain}/{len(healthy)} ({healthy_abstain/len(healthy)*100:.1f}%)")
