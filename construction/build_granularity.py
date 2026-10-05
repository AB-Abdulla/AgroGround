"""
Granularity assignment for AgroGround (construction step 4).

Reads the detector output of every groundable sample (``final_gdino_results.jsonl``, produced by ``final_gdino_script.py``) and the preserved non-groundable samples (``final_nongroundable_samples.jsonl``, produced by ``capture_nongroundable.py``), and writes one reorganized file in which every record carries an explicit ``granularity_level``:

    fine_region is lesion-level boxes: every surviving box covering <= AREA_CEILING of the image, after per-group NMS
    coarse_object is object-level box: the single highest-confidence box covering > AREA_CEILING of the image, after per-group NMS
    non_groundable is no box (healthy image, plant identification, management action, cause, ...), the routing reason is carried through
    detection_failure the sample is groundable but the detector returned no box, recorded for reporting, excluded from training

A groundable sample whose boxes fall on both sides of AREA_CEILING yields one fine_region record and one coarse_object record (each later receives its own instruction), which is how the corpus reaches 391,758 lesion-level and 403,092 object-level examples from 579,260 boxed samples.

Filtering order: boxes below MIN_BOX_FRAC of the image are dropped, boxes are split by area into small/large groups, and NMS at NMS_IOU is applied within each group (so a large box never suppresses a small one). Boxes stay in pixel coordinates, normalization to the 0--1000 grid happens in ``convert_to_qwen.py``.

Inputs (under AGROGROUND_WORK_DIR): final_gdino_results.jsonl, final_nongroundable_samples.jsonl
Outputs (under AGROGROUND_WORK_DIR): agroground_granularity.jsonl, agroground_granularity_summary.json

Configuration is read from two environment variables:
    AGROGROUND_WORK_DIR: directory holding the pipeline's intermediate files (default ./work)
    AGROGROUND_DATA_ROOT: directory holding the eight source datasets, one sub-folder each (default ./datasets); see README for the expected layout
"""
import os
import json
import time
from collections import Counter, defaultdict

from PIL import Image

WORK_DIR = os.environ.get("AGROGROUND_WORK_DIR", "./work")
DATA_ROOT = os.environ.get("AGROGROUND_DATA_ROOT", "./datasets")

GROUNDABLE = os.path.join(WORK_DIR, "final_gdino_results.jsonl")
NONGROUND = os.path.join(WORK_DIR, "final_nongroundable_samples.jsonl")
OUTPUT = os.path.join(WORK_DIR, "agroground_granularity.jsonl")
SUMMARY = os.path.join(WORK_DIR, "agroground_granularity_summary.json")

AREA_CEILING = 0.50   # boxes <= this fraction of the image are lesion-level, larger boxes are object-level
NMS_IOU = 0.50        # within a group, boxes overlapping a kept box above this IoU are duplicates
MIN_BOX_FRAC = 0.001  # boxes smaller than this fraction of the image are dropped as degenerate

# The CDDM release stores image paths under its authors' own prefix, map it to the local copy.
PATH_REMAP = {
    "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv": os.path.join(DATA_ROOT, "cddm", "images"),
}


def resolve_image_path(p):
    """Map a stored image path to a local file: apply source-specific prefix remaps,
    then treat relative paths as relative to AGROGROUND_DATA_ROOT."""
    if not p:
        return p
    for src, dst in PATH_REMAP.items():
        if p.startswith(src):
            return dst + p[len(src):]
    if not os.path.isabs(p):
        return os.path.join(DATA_ROOT, p)
    return p


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = aa + ba - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, confs, thr):
    """Standard NMS: keep the highest-confidence box, drop boxes overlapping it above thr, repeat."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: -confs[i])
    keep = []
    while order:
        cur = order.pop(0)
        keep.append(cur)
        order = [i for i in order if iou(boxes[cur], boxes[i]) <= thr]
    return keep


def clean_group(boxes, confs):
    """Apply NMS to one group of boxes and return the surviving boxes."""
    idx = nms(boxes, confs, NMS_IOU)
    return [boxes[i] for i in idx]


def main():
    print("Loading non-groundable samples")
    nonground = []
    with open(NONGROUND) as f:
        for line in f:
            try:
                nonground.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  {len(nonground):,} non-groundable")

    print("Processing groundable samples (reading image sizes)")
    counts = Counter()
    per_gran = Counter()
    per_dataset_gran = defaultdict(Counter)
    both_count = 0
    t0 = time.time()

    os.makedirs(WORK_DIR, exist_ok=True)
    fout = open(OUTPUT, "w")

    n = 0
    with open(GROUNDABLE) as f:
        for line in f:
            n += 1
            if n % 50000 == 0:
                print(f"  {n:,} processed ({(time.time() - t0) / 60:.1f} min)")
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts["groundable_read"] += 1

            boxes = r.get("stageE_boxes") or []
            confs = r.get("stageE_confidences") or [1.0] * len(boxes)
            if len(confs) < len(boxes):
                confs = confs + [0.0] * (len(boxes) - len(confs))

            # Groundable sample on which the detector returned nothing.
            if not r.get("stageE_hit") or not boxes:
                rec = {
                    "sample_id": r.get("sample_id"),
                    "image_path": r.get("image_path"),
                    "granularity_level": "detection_failure",
                    "boxes": [],
                    "stageD_kind": r.get("stageD_kind"),
                    "stageD_label": r.get("stageD_label"),
                    "stageD_crop": r.get("stageD_crop"),
                    "stageD_dataset": r.get("stageD_dataset"),
                    "stageE_prompt": r.get("stageE_prompt"),
                    "source_hit": False,
                }
                fout.write(json.dumps(rec) + "\n")
                counts["detection_failure"] += 1
                per_gran["detection_failure"] += 1
                continue

            # Image size is needed for the area fractions.
            img_path = resolve_image_path(r.get("image_path", ""))
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except (OSError, ValueError):
                counts["image_error"] += 1
                continue
            area = W * H
            if area == 0:
                counts["image_error"] += 1
                continue

            # Area split first, NMS afterwards within each group.
            small, small_c, big, big_c = [], [], [], []
            for b, c in zip(boxes, confs):
                frac = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) / area
                if frac < MIN_BOX_FRAC:
                    continue
                if frac <= AREA_CEILING:
                    small.append(b)
                    small_c.append(c)
                else:
                    big.append(b)
                    big_c.append(c)

            ds = r.get("stageD_dataset")
            emitted = []

            if small:
                fine_boxes = clean_group(small, small_c)
                if fine_boxes:
                    rec = {
                        "sample_id": f'{r.get("sample_id")}_fine',
                        "source_sample_id": r.get("sample_id"),
                        "image_path": r.get("image_path"),
                        "image_width": W, "image_height": H,
                        "granularity_level": "fine_region",
                        "boxes": fine_boxes,
                        "stageD_kind": r.get("stageD_kind"),
                        "stageD_label": r.get("stageD_label"),
                        "stageD_crop": r.get("stageD_crop"),
                        "stageD_dataset": ds,
                        "stageE_prompt": r.get("stageE_prompt"),
                    }
                    fout.write(json.dumps(rec) + "\n")
                    per_gran["fine_region"] += 1
                    per_dataset_gran[ds]["fine_region"] += 1
                    emitted.append("fine")

            if big:
                coarse_boxes = clean_group(big, big_c)
                if coarse_boxes:
                    # The single highest-confidence large box is the object-level box.
                    rec = {
                        "sample_id": f'{r.get("sample_id")}_coarse',
                        "source_sample_id": r.get("sample_id"),
                        "image_path": r.get("image_path"),
                        "image_width": W, "image_height": H,
                        "granularity_level": "coarse_object",
                        "boxes": [coarse_boxes[0]],
                        "stageD_kind": r.get("stageD_kind"),
                        "stageD_label": r.get("stageD_label"),
                        "stageD_crop": r.get("stageD_crop"),
                        "stageD_dataset": ds,
                        "stageE_prompt": r.get("stageE_prompt"),
                    }
                    fout.write(json.dumps(rec) + "\n")
                    per_gran["coarse_object"] += 1
                    per_dataset_gran[ds]["coarse_object"] += 1
                    emitted.append("coarse")

            if "fine" in emitted and "coarse" in emitted:
                both_count += 1
            if not emitted:
                counts["emptied_after_filter"] += 1

    print("Writing non-groundable samples...")
    for s in nonground:
        rec = {
            "sample_id": s.get("sample_id"),
            "image_path": s.get("image_path"),
            "granularity_level": "non_groundable",
            "boxes": [],
            "stageD_kind": s.get("stageD_kind"),
            "stageD_label": s.get("stageD_label"),
            "stageD_crop": s.get("stageD_crop"),
            "stageD_dataset": s.get("stageD_dataset"),
            "nongroundable_reason": s.get("nongroundable_reason"),
            "question": s.get("question", ""),
            "answer": s.get("answer", ""),
        }
        fout.write(json.dumps(rec) + "\n")
        per_gran["non_groundable"] += 1

    fout.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}\nRESULTS\n{'=' * 60}")
    print(f"Runtime: {elapsed / 60:.1f} min")
    print(f"\nGroundable read      : {counts['groundable_read']:,}")
    print(f"Image errors         : {counts['image_error']:,}")
    print(f"Emptied after filter : {counts['emptied_after_filter']:,}")
    print("\nGranularity levels (records written):")
    for g, c in per_gran.most_common():
        print(f"  {g:<20}: {c:>10,}")
    print(f"\nImages emitting both a fine and a coarse record: {both_count:,}")
    print(f"\nTotal records written: {sum(per_gran.values()):,}")
    print("\nPer dataset (fine / coarse):")
    for ds in sorted(per_dataset_gran.keys()):
        d = per_dataset_gran[ds]
        print(f"  {ds:<12}: fine={d.get('fine_region', 0):>8,}  coarse={d.get('coarse_object', 0):>8,}")

    summary = {
        "area_ceiling": AREA_CEILING, "nms_iou": NMS_IOU, "min_box_frac": MIN_BOX_FRAC,
        "runtime_min": round(elapsed / 60, 1),
        "groundable_read": counts["groundable_read"],
        "image_errors": counts["image_error"],
        "granularity_counts": dict(per_gran),
        "both_images": both_count,
        "total_examples": sum(per_gran.values()),
        "per_dataset": {ds: dict(d) for ds, d in per_dataset_gran.items()},
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutput: {OUTPUT}")
    print(f"Summary: {SUMMARY}\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
