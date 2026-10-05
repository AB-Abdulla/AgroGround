
"""
GroundingDINO baseline on the benchmark with raw disease names.

Runs GroundingDINO on the 1,480 benchmark images with the same configuration as
the annotation run (box threshold 0.25, text threshold 0.20), prompted with the
raw disease name of each image (for healthy images, the crop-matched absent
disease; finding nothing is the correct output). Boxes are written in pixel
coordinates in the standard prediction format. This is the "raw prompts" teacher
configuration in the paper's tables; the annotator configuration (taxonomy
prompts plus the pipeline's geometry filters) is scored from the taxonomy run by
score.py.

Input: human_verified_ground_truth_v4.jsonl in the work directory.
Output: gdino_given_v4.json.
"""
import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import os, sys, json, time
import torch

WEIGHTS_DIR   = "./weights"
GDINO_CONFIG  = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GT_FILE = "./work/human_verified_ground_truth_v4.jsonl"
OUTPUT  = "./work/gdino_given_v4.json"

BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.20

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

print("[Model] Loading GroundingDINO...")
from groundingdino.util.inference import load_model, load_image, predict
model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
print("[Model] Loaded")

# Load ground truth eval set
gt = []
with open(GT_FILE) as f:
    for line in f:
        g = json.loads(line)
        if g['granularity'] in ('fine','coarse','healthy'):
            gt.append(g)
print(f"Running GDino on {len(gt)} images...")

def get_raw_prompt(g):
    if g['granularity'] == 'healthy':
        return g.get('abstention_disease') or 'disease'
    return (g.get('abstention_disease') if g.get('granularity')=='healthy' else g.get('disease_raw')) or 'disease'

results = []
t0 = time.time()
for i, g in enumerate(gt):
    if i % 50 == 0 and i > 0:
        el = time.time()-t0
        print(f"  {i}/{len(gt)} ({el/60:.1f} min, ETA {el/i*(len(gt)-i)/60:.0f} min)")
    img_path = fix_path(g['orig_path'])
    prompt = get_raw_prompt(g).lower().strip()
    W, H = g['image_width'], g['image_height']
    pred_boxes = []
    try:
        image_source, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=model, image=image, caption=prompt,
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE)
        # GDino returns normalized cxcywh (0-1). Convert to pixel x1y1x2y2.
        Hs, Ws = image_source.shape[:2]
        for b in boxes:
            cx, cy, bw, bh = b.tolist()
            x1 = (cx - bw/2) * Ws; y1 = (cy - bh/2) * Hs
            x2 = (cx + bw/2) * Ws; y2 = (cy + bh/2) * Hs
            pred_boxes.append([round(x1), round(y1), round(x2), round(y2)])
    except Exception as e:
        print(f"  GDino error on {img_path}: {e}")

    results.append({
        'coco_id': g['coco_id'], 'granularity': g['granularity'], 'dataset': g['dataset'],
        'gt_boxes': g['gt_boxes'],           # pixel x1y1x2y2 (human)
        'pred_boxes_pixel': pred_boxes,      # pixel x1y1x2y2 (GDino)
        'image_width': W, 'image_height': H,
        'prompt': prompt,
    })

with open(OUTPUT, "w") as f:
    json.dump(results, f)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved {len(results)} predictions.")
print(f"Saved: {OUTPUT}")
