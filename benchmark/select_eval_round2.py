"""
Round 2 candidate selection for the human-verified benchmark.

Selects 740 new images to extend the benchmark after round 1:
  dense_fine 250   lesion-level images with at least 5 detector boxes
  fine 150   further lesion-level images (fewer than 5 boxes)
  coarse 70   object-level images
  healthy 270   healthy images with crop metadata (needed for crop-matched disease queries)

Rules: positives come from the held-out test split only; every image of the
round-1 manifest is excluded; healthy images must carry a crop; CDDM is capped
at about 40% of each stratum and the rest is allocated by availability.
Selection is seeded (seed 31).

Output, in the work directory: eval_round2_manifest.jsonl (same schema as round 1)
and eval_healthy_paths_all.txt, the list of all healthy benchmark images from
rounds 1 and 2, which the negatives builders exclude from the healthy pool.
"""
import json, random
from collections import defaultdict, Counter
from PIL import Image

random.seed(31)

GRAN_FILE = "./work/agroground_granularity.jsonl"
NONGROUND = "./work/final_nongroundable_samples.jsonl"
TEST_SPLIT = "./work/agroground_test.jsonl"
ROUND1 = "./work/eval_1000_manifest.jsonl"
OUTPUT = "./work/eval_round2_manifest.jsonl"
LOCKFILE = "./work/eval_healthy_paths_all.txt"

STRATUM_TARGETS = {"dense_fine": 250, "fine": 150, "coarse": 70, "healthy": 270}
CDDM_CAP_FRAC = 0.40
DENSE_MIN_BOXES = 5

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

print("Loading round-1 manifest for exclusion...")
r1_paths = set(); r1_healthy_paths = []
with open(ROUND1) as f:
    for line in f:
        m = json.loads(line)
        fp = fix_path(m["image_path"])
        r1_paths.add(fp)
        if m.get("regime") == "target_absent":
            r1_healthy_paths.append(fp)
print(f"  round-1 images excluded: {len(r1_paths)} (healthy among them: {len(r1_healthy_paths)})")

print("Loading test-split source IDs...")
test_ids = set()
with open(TEST_SPLIT) as f:
    for line in f:
        test_ids.add(json.loads(line).get("source_sample_id"))

print("Scanning positives (test split, not in round 1)...")
pools = {"dense_fine": defaultdict(list), "fine": defaultdict(list), "coarse": defaultdict(list)}
with open(GRAN_FILE) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get("source_sample_id") not in test_ids: continue
        if fix_path(r.get("image_path","")) in r1_paths: continue
        g = r.get("granularity_level"); ds = r.get("stageD_dataset")
        nb = len(r.get("boxes") or [])
        if g == "fine_region":
            pools["dense_fine" if nb >= DENSE_MIN_BOXES else "fine"][ds].append(r)
        elif g == "coarse_object":
            pools["coarse"][ds].append(r)

print("Scanning healthy (crop metadata required, not in round 1)...")
no_crop = 0
pools["healthy"] = defaultdict(list)
with open(NONGROUND) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get("nongroundable_reason") != "healthy": continue
        if fix_path(r.get("image_path","")) in r1_paths: continue
        crop = (r.get("stageD_crop") or "").strip()
        if not crop:
            no_crop += 1; continue
        pools["healthy"][r.get("stageD_dataset")].append(r)
print(f"  healthy without crop metadata skipped: {no_crop:,}")

for stratum in pools:
    avail = {ds: len(v) for ds, v in pools[stratum].items()}
    print(f"  pool[{stratum}]: total {sum(avail.values()):,} | " +
          ", ".join(f"{d}={n}" for d, n in sorted(avail.items(), key=lambda x: -x[1])))

def allocate(stratum, target):
    """CDDM capped at CDDM_CAP_FRAC of target; others filled by availability."""
    by_ds = pools[stratum]
    chosen = []
    cddm_cap = int(target * CDDM_CAP_FRAC)
    take_cddm = min(cddm_cap, len(by_ds.get("cddm", [])))
    chosen += random.sample(by_ds.get("cddm", []), take_cddm) if take_cddm else []
    remaining = target - len(chosen)
    others = {ds: list(v) for ds, v in by_ds.items() if ds != "cddm"}
    # round-robin by availability until filled or exhausted
    ds_order = sorted(others, key=lambda d: -len(others[d]))
    for ds in ds_order: random.shuffle(others[ds])
    while remaining > 0 and any(others.values()):
        for ds in ds_order:
            if remaining <= 0: break
            if others[ds]:
                chosen.append(others[ds].pop()); remaining -= 1
    # if still short (small pools), top up from cddm
    if remaining > 0:
        extra_pool = [r for r in by_ds.get("cddm", []) if r not in chosen]
        top = random.sample(extra_pool, min(remaining, len(extra_pool)))
        chosen += top; remaining -= len(top)
    return chosen

def get_wh(path):
    try:
        with Image.open(fix_path(path)) as im: return im.size
    except: return None, None

manifest = []
report = defaultdict(Counter)
for stratum, target in STRATUM_TARGETS.items():
    picked = allocate(stratum, target)
    for r in picked:
        ds = r.get("stageD_dataset")
        if stratum == "healthy":
            W, H = get_wh(r.get("image_path",""))
            manifest.append({"image_path": r["image_path"], "regime": "target_absent",
                "granularity": "non_groundable", "dataset": ds,
                "disease_query": None, "raw_label": r.get("stageD_label"),
                "crop": (r.get("stageD_crop") or "").strip(),
                "gdino_boxes": [], "image_width": W, "image_height": H,
                "stratum": stratum})
        else:
            regime = "single_target" if stratum == "coarse" else "multi_target"
            gran = "coarse_object" if stratum == "coarse" else "fine_region"
            manifest.append({"image_path": r["image_path"], "regime": regime,
                "granularity": gran, "dataset": ds,
                "disease_query": r.get("stageE_prompt"), "raw_label": r.get("stageD_label"),
                "crop": (r.get("stageD_crop") or "").strip(),
                "gdino_boxes": r.get("boxes"),
                "image_width": r.get("image_width"), "image_height": r.get("image_height"),
                "stratum": stratum})
        report[stratum][ds] += 1

random.shuffle(manifest)
with open(OUTPUT, "w") as f:
    for m in manifest:
        f.write(json.dumps(m) + "\n")

# lockfile: ALL eval healthy paths (round 1 + round 2) for the negatives-SFT builder
r2_healthy = [fix_path(m["image_path"]) for m in manifest if m["regime"] == "target_absent"]
with open(LOCKFILE, "w") as f:
    for p in r1_healthy_paths + r2_healthy:
        f.write(p + "\n")

print(f"\n{'='*64}\nROUND-2 SELECTION COMPLETE\n{'='*64}")
print(f"Total selected: {len(manifest)}  (target {sum(STRATUM_TARGETS.values())})")
for stratum in STRATUM_TARGETS:
    line = ", ".join(f"{d}={c}" for d, c in sorted(report[stratum].items(), key=lambda x: -x[1]))
    print(f"  {stratum:<11} {sum(report[stratum].values()):>4}: {line}")
print(f"\nManifest : {OUTPUT}")
print(f"Lockfile : {LOCKFILE}  ({len(r1_healthy_paths)+len(r2_healthy)} eval-healthy paths locked out of training)")
