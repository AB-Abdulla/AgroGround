r"""
Converts the granularity-labeled records into Qwen3-VL training examples (construction step 5).

Input: agroground_granularity.jsonl (from build_granularity.py).
Only fine_region and coarse_object records are converted. Non-groundable and
detection-failure records are skipped here. Negative examples are added later by
build_negatives_12pct.py and build_negatives_6pct.py.

Each example is an instruction plus a JSON answer:
  - The instruction is picked (seed 42) from four lesion-level templates or four
    object-level templates, so the wording tells the model which granularity to
    produce. The {p} placeholder is filled with the sample's taxonomy prompt
    (stageE_prompt), which is the same crop-aware prompt that produced the boxes.
  - The answer is a JSON list of {"bbox_2d": [x1, y1, x2, y2], "label": label}.
    Coordinates are normalized to the 0-1000 grid (x/W*1000, y/H*1000), which is
    the native Qwen3-VL box format. The label is the normalized target label
    (stageD_label).

Output: agroground_qwen_all.jsonl in the work directory, one record per example:
  {"image": image path,
   "conversations": [{"from": "human", "value": "<image>\n<instruction>"},
                     {"from": "gpt", "value": "[{\"bbox_2d\": [...], \"label\": \"...\"}, ...]"}],
   "granularity_level": ..., "dataset": ..., "source_sample_id": ...}

The 80/10/10 split is done afterwards by split_train_val_test.py.

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
"""
import json, random, re

INPUT   = "./work/agroground_granularity.jsonl"
OUTPUT  = "./work/agroground_qwen_all.jsonl"
random.seed(42)

# ── Instruction templates (varied phrasings so model generalizes, not memorizes)
FINE_TEMPLATES = [
    "Locate the {p} in this image. Output the bounding boxes of the affected regions in JSON format.",
    "Detect the specific lesions of {p} and return their bounding boxes in JSON format.",
    "Identify the individual affected areas ({p}) on the leaf. Output bounding boxes in JSON.",
    "Find the disease lesions ({p}) in the image and output their bounding boxes in JSON format.",
]
COARSE_TEMPLATES = [
    "Locate the leaf affected by {p}. Output its bounding box in JSON format.",
    "Detect the affected leaf ({p}) and return its bounding box in JSON format.",
    "Identify the affected plant region ({p}) in this image. Output the bounding box in JSON.",
    "Find the diseased leaf ({p}) in the image and output its bounding box in JSON format.",
]

def norm_box(box, W, H):
    """Normalize pixel box to 0-1000 (Qwen3-VL convention). Clamp to [0,1000]."""
    x1, y1, x2, y2 = box
    nx1 = max(0, min(round(x1 / W * 1000), 1000))
    ny1 = max(0, min(round(y1 / H * 1000), 1000))
    nx2 = max(0, min(round(x2 / W * 1000), 1000))
    ny2 = max(0, min(round(y2 / H * 1000), 1000))
    return [nx1, ny1, nx2, ny2]

def short_label(prompt, label):
    """A concise label for the bbox_2d 'label' field. Use the disease label."""
    if label and label.strip():
        return label.strip().lower()
    return (prompt or "affected region").strip().lower()

counts = {"fine_region": 0, "coarse_object": 0, "skipped": 0}
n_in = 0

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        n_in += 1
        try:
            r = json.loads(line)
        except:
            continue
        g = r.get("granularity_level")
        if g not in ("fine_region", "coarse_object"):
            continue  # skip non_groundable and detection_failure for Phase 1

        W = r.get("image_width"); H = r.get("image_height")
        boxes = r.get("boxes") or []
        if not W or not H or not boxes:
            counts["skipped"] += 1
            continue

        prompt = (r.get("stageE_prompt") or "").strip()
        label  = short_label(prompt, r.get("stageD_label"))

        # normalize boxes
        nboxes = [norm_box(b, W, H) for b in boxes]

        # build the assistant answer: JSON list of bbox_2d + label
        answer = json.dumps([{"bbox_2d": nb, "label": label} for nb in nboxes])

        # pick instruction template by granularity (rotate through variants)
        if g == "fine_region":
            tmpl = random.choice(FINE_TEMPLATES)
        else:
            tmpl = random.choice(COARSE_TEMPLATES)
        instruction = tmpl.format(p=prompt)

        record = {
            "image": r.get("image_path"),
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt",   "value": answer},
            ],
            # keep metadata for the split + later analysis (Qwen ignores extra keys)
            "granularity_level": g,
            "dataset": r.get("stageD_dataset"),
            "source_sample_id": r.get("source_sample_id"),
        }
        fout.write(json.dumps(record) + "\n")
        counts[g] += 1

print("="*60)
print("CONVERSION COMPLETE")
print("="*60)
print(f"Input lines read : {n_in:,}")
print(f"fine_region      : {counts['fine_region']:,}")
print(f"coarse_object    : {counts['coarse_object']:,}")
print(f"skipped (no box/dims): {counts['skipped']:,}")
print(f"Total converted  : {counts['fine_region']+counts['coarse_object']:,}")
print(f"Output: {OUTPUT}")
