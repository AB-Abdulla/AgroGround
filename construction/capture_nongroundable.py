"""
Non-groundable stream capture for AgroGround (construction step 2).

final_preprocess.py keeps only the groundable samples (the 615,485 whose answers name a visually localizable target). This script re-runs the same source-specific handlers, blocklist, and routing rules on the same inputs, but writes the samples that were routed out, so that the full corpus accounting is preserved and the healthy stream is available for negative examples.

Output (work directory): final_nongroundable_samples.jsonl
    One record per non-groundable sample, with nongroundable_reason:
        healthy: healthy image, no target present
        action: the answer is a management action
        cause: the answer is a cause or condition, not a visible target
        plant_id: plant species identification (not groundable)
        aerial_or_blocklist: aerial field imagery or a blocklisted label
        handler_none: the handler could not parse the sample

Run after final_preprocess.py; both must see identical inputs so that the groundable and non-groundable partitions sum to the ingested total.

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
"""
import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from stageA_label_counts import (
    handle_agrobench_sample, handle_agrocot_sample, handle_cddm_sample,
    handle_agmmu_sample, handle_agromind_sample, handle_leafbench_sample,
    handle_leafnet_sample, handle_mirage_sample,
)

JSONL_DIR   = "./work/unified_jsonl"
OUTPUT_PATH = "./work/final_nongroundable_samples.jsonl"
SUMMARY_PATH = "./work/final_nongroundable_summary.json"

DATASET_HANDLERS = {
    "agrobench": handle_agrobench_sample,
    "agrocot":   handle_agrocot_sample,
    "cddm":      handle_cddm_sample,
    "agmmu":     handle_agmmu_sample,
    "agromind":  handle_agromind_sample,
    "leafbench": handle_leafbench_sample,
    "leafnet":   handle_leafnet_sample,
    "mirage":    handle_mirage_sample,
}

# EXACT same rules as final_preprocess.py
NOT_GROUNDABLE = {"action", "cause"}
LABEL_BLOCKLIST = {
    'weed_cluster', 'double_plant', 'cloud_shadow',
    'standing_water', 'waterway', 'planter_skip',
    'raccoon', 'ball python', 'fungus', 'plant',
    'apple', 'black pepper', 'cashew', 'cassava', 'cherry',
    'coffee', 'cucumber', 'grape', 'maize', 'mango', 'orange',
    'peach', 'pepper', 'potato', 'raspberry', 'rice', 'soybean',
    'squash', 'strawberry', 'sugarcane', 'tea', 'tomato',
}

def fix_image_path(path):
    if path and path.startswith("/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv"):
        return path.replace(
            "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv",
            "./datasets/cddm/images", 1)
    return path

def get_image_path(sample):
    path = sample.get("image_path")
    if not path:
        views = sample.get("views") or []
        path = views[0] if views else None
    return fix_image_path(path) if path else None

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}\nCapture Non-Groundable | Run {run_id}\n{'='*60}\n")

    counts = Counter()
    per_reason = Counter()
    per_dataset = defaultdict(Counter)
    t0 = time.time()

    with open(OUTPUT_PATH, "w") as fout:
        for ds_name, handler in DATASET_HANDLERS.items():
            jsonl_path = os.path.join(JSONL_DIR, f"{ds_name}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"[{ds_name}] File not found, skipping")
                continue
            ds_captured = 0
            with open(jsonl_path) as f:
                for line in tqdm(f, desc=f"{ds_name:<12}", unit=" samples"):
                    counts["total_read"] += 1
                    try:
                        sample = json.loads(line)
                    except:
                        continue
                    sample_id = sample.get("sample_id", "")

                    # Run handler, determine reason for non-groundability
                    result = handler(sample)
                    reason = None

                    if result is None:
                        reason = "handler_none"
                    else:
                        kind  = result.get("kind", "")
                        label = result.get("label", "")
                        if kind in NOT_GROUNDABLE:
                            reason = kind  # "action" or "cause"
                        elif ds_name == 'cddm' and kind == 'object':
                            reason = "plant_id"
                        elif label.lower().strip() in LABEL_BLOCKLIST:
                            reason = "aerial_or_blocklist"
                        elif kind == "healthy":
                            reason = "healthy"
                        # else: it's groundable then we do NOT capture it here

                    if reason is None:
                        continue  # groundable sample, skip (already in final_samples.jsonl)

                    # Capture this non-groundable sample
                    kind  = result.get("kind", "") if result else ""
                    label = result.get("label", "") if result else ""
                    crop  = result.get("crop") if result else None
                    image_path = get_image_path(sample)

                    record = {
                        "sample_id":            sample_id,
                        "source":               ds_name,
                        "image_path":           image_path,
                        "question":             sample.get("question", ""),
                        "answer":               sample.get("answer", ""),
                        "metadata":             sample.get("metadata", {}),
                        "stageD_kind":          kind,
                        "stageD_label":         label,
                        "stageD_crop":          crop,
                        "stageD_dataset":       ds_name,
                        "nongroundable_reason": reason,
                    }
                    fout.write(json.dumps(record) + "\n")
                    counts["captured"] += 1
                    per_reason[reason] += 1
                    per_dataset[ds_name][reason] += 1
                    ds_captured += 1
            print(f"  {ds_name}: {ds_captured:,} non-groundable captured")

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Runtime: {elapsed}s")
    print(f"Total read     : {counts['total_read']:,}")
    print(f"Total captured : {counts['captured']:,}")
    print(f"\nBy reason:")
    for r, c in per_reason.most_common():
        print(f"  {r:<22}: {c:>10,}")
    print(f"\nBy dataset:")
    for ds in DATASET_HANDLERS:
        if ds in per_dataset:
            tot = sum(per_dataset[ds].values())
            parts = ", ".join(f"{k}:{v}" for k,v in sorted(per_dataset[ds].items()))
            print(f"  {ds:<12}: {tot:>8,}  ({parts})")

    summary = {
        "run_id": run_id, "runtime_s": elapsed,
        "total_read": counts["total_read"], "captured": counts["captured"],
        "by_reason": dict(per_reason),
        "by_dataset": {ds: dict(r) for ds, r in per_dataset.items()},
        "output_path": OUTPUT_PATH,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}\n{'='*60}\n")

if __name__ == "__main__":
    main()
