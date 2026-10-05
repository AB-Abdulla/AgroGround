"""
Round 1 candidate selection for the human-verified benchmark.

Selects 1,000 images for annotation, stratified over the three evaluation regimes and
over the eight sources (proportional, with a minimum per source):
  fine_region     multi-target images (lesion sets)     500
  coarse_object   single-target images (whole leaf)     300
  target_absent   healthy images (correct output: none) 200

Positive candidates are drawn from the held-out test split, which is a split by
source sample; because the sources reuse photographs across samples, this round did
not yet exclude images whose files also occur in the training split. That rule was
introduced in round 3, and the perceptual-hash audit later removed the training
copies this round had admitted (see the paper's benchmark section). Healthy candidates
come from the healthy stream. Positive images carry the detector's boxes as editable
pre-annotations; healthy images have an empty target. Selection is seeded (seed 2024).

Input: agroground_granularity.jsonl, final_nongroundable_samples.jsonl and
agroground_test.jsonl in the work directory. Output: eval_1000_manifest.jsonl, one
record per image with image_path, regime, granularity, dataset, disease_query,
gdino_boxes (pixel coordinates) and image width and height.
"""
import json, random
from collections import defaultdict, Counter
from PIL import Image

random.seed(2024)

GRAN_FILE   = "./work/agroground_granularity.jsonl"
NONGROUND   = "./work/final_nongroundable_samples.jsonl"
TEST_SPLIT  = "./work/agroground_test.jsonl"   # to restrict to test-set images only
OUTPUT      = "./work/eval_1000_manifest.jsonl"

N_FINE, N_COARSE, N_ABSENT = 500, 300, 200

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

# --- Restrict evaluation to images that were in the TEST split (never trained on) ---
print("Loading test-split source image IDs (so eval images were never trained on)...")
test_source_ids = set()
with open(TEST_SPLIT) as f:
    for line in f:
        try:
            r = json.loads(line)
            test_source_ids.add(r.get("source_sample_id"))
        except:
            continue
print(f"  {len(test_source_ids):,} unique test-split source images")

# --- Load positive granularity examples, keep only those from the test split ---
print("Loading positive (fine/coarse) examples from test split...")
fine_by_ds = defaultdict(list)
coarse_by_ds = defaultdict(list)
with open(GRAN_FILE) as f:
    for line in f:
        try:
            r = json.loads(line)
        except:
            continue
        g = r.get("granularity_level")
        sid = r.get("source_sample_id")
        if sid not in test_source_ids:
            continue
        ds = r.get("stageD_dataset")
        if g == "fine_region":
            fine_by_ds[ds].append(r)
        elif g == "coarse_object":
            coarse_by_ds[ds].append(r)

# --- Load healthy (target-absent) samples ---
print("Loading healthy (target-absent) samples...")
healthy_by_ds = defaultdict(list)
with open(NONGROUND) as f:
    for line in f:
        try:
            r = json.loads(line)
        except:
            continue
        if r.get("nongroundable_reason") == "healthy":
            healthy_by_ds[r.get("stageD_dataset")].append(r)

def stratified_pick(by_ds, n_total, min_per_ds=5):
    """Pick n_total items stratified across datasets, proportional with a floor."""
    datasets = [ds for ds in by_ds if by_ds[ds]]
    total_avail = sum(len(by_ds[ds]) for ds in datasets)
    picked = []
    # proportional allocation
    for ds in datasets:
        share = len(by_ds[ds]) / total_avail
        n_ds = max(min_per_ds, round(share * n_total))
        n_ds = min(n_ds, len(by_ds[ds]))
        picked.extend(random.sample(by_ds[ds], n_ds))
    # trim or pad to exactly n_total
    random.shuffle(picked)
    if len(picked) > n_total:
        picked = picked[:n_total]
    return picked

fine_pick   = stratified_pick(fine_by_ds, N_FINE)
coarse_pick = stratified_pick(coarse_by_ds, N_COARSE)
absent_pick = stratified_pick(healthy_by_ds, N_ABSENT)

print(f"\nPicked: fine={len(fine_pick)}, coarse={len(coarse_pick)}, absent={len(absent_pick)}")

# --- Build the manifest ---
def get_wh(path):
    try:
        with Image.open(fix_path(path)) as im:
            return im.size
    except:
        return None, None

manifest = []
for r in fine_pick:
    manifest.append({
        "image_path": r["image_path"], "regime": "multi_target",
        "granularity": "fine_region", "dataset": r["stageD_dataset"],
        "disease_query": r.get("stageE_prompt"), "raw_label": r.get("stageD_label"),
        "gdino_boxes": r.get("boxes"),  # pre-annotations (pixel coords)
        "image_width": r.get("image_width"), "image_height": r.get("image_height"),
    })
for r in coarse_pick:
    manifest.append({
        "image_path": r["image_path"], "regime": "single_target",
        "granularity": "coarse_object", "dataset": r["stageD_dataset"],
        "disease_query": r.get("stageE_prompt"), "raw_label": r.get("stageD_label"),
        "gdino_boxes": r.get("boxes"),
        "image_width": r.get("image_width"), "image_height": r.get("image_height"),
    })
for r in absent_pick:
    W, H = get_wh(r.get("image_path",""))
    manifest.append({
        "image_path": r["image_path"], "regime": "target_absent",
        "granularity": "non_groundable", "dataset": r["stageD_dataset"],
        "disease_query": None, "raw_label": r.get("stageD_label"),  # healthy: no disease
        "gdino_boxes": [],  # empty ground truth
        "image_width": W, "image_height": H,
    })

random.shuffle(manifest)
with open(OUTPUT, "w") as f:
    for m in manifest:
        f.write(json.dumps(m) + "\n")

# --- Summary ---
print(f"\n{'='*60}\nEVAL SET MANIFEST BUILT\n{'='*60}")
print(f"Total images: {len(manifest)}")
regime_counts = Counter(m["regime"] for m in manifest)
print("\nBy regime:")
for r, c in regime_counts.items():
    print(f"  {r:<16}: {c}")
print("\nBy dataset:")
ds_counts = Counter(m["dataset"] for m in manifest)
for ds, c in sorted(ds_counts.items()):
    print(f"  {ds:<12}: {c}")
print(f"\nOutput: {OUTPUT}")
