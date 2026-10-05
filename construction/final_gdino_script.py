"""
Weak box and mask annotation for AgroGround (construction step 3).

Runs every groundable sample (final_samples.jsonl, from final_preprocess.py) through
GroundingDINO and SAM2 and writes final_gdino_results.jsonl, one record per sample,
resumable if interrupted.

For each sample the prompt is read from stageB_taxonomy.json (crop-aware, with symptom
cues; the synonym map stageA2_synonym_map.json is applied, both files ship next to this
script). If a label has no taxonomy entry, the label itself is used. GroundingDINO runs
at box threshold 0.25 and text threshold 0.20. Boxes are then filtered by geometry:
boxes smaller than 0.1% of the image or larger than 95% of it are dropped, degenerate
boxes are dropped, and exact duplicate boxes are merged. For multi-instance targets
(diseases, pests, weeds, symptoms) all remaining boxes are kept; for single-object
targets the highest-confidence box is kept. The kind of target is taken from stageD_kind.
SAM2 then produces a mask from the kept boxes; the mask file is saved only when its
combined coverage lies between 0.1% and 95% of the image. The mask does not filter the
boxes: every kept box is written whether or not a mask file was saved.

Each output record contains the original sample fields plus:
    stageE_prompt            the prompt used
    stageE_prompt_source     where the prompt came from (taxonomy, lookup, or label)
    stageE_boxes             list of [x1, y1, x2, y2] pixel boxes
    stageE_confidences       detector confidence per box
    stageE_mask_path         path of the saved mask, or null
    stageE_runtime_seconds   processing time for the sample
    stageE_gpu_memory_mb     peak GPU memory after the sample
    stageE_model             "gdino+sam2"
    stageE_hit               true if at least one box survived
    stageE_skipped_reason    no_image, no_boxes or no_prompt, when skipped

Image paths stored by the CDDM release are mapped to the local copy automatically.
Paths are relative to the repository root: intermediate files under ./work, source datasets under ./datasets (one sub-folder per source). GroundingDINO and SAM2 weights are read from ./weights.
"""

import sys
sys.path = [p for p in sys.path if '/sam2/sam2' not in p]

import os
import json
import time
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLES_PATH   = "./work/final_samples.jsonl"
TAXONOMY_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stageB_taxonomy.json")
OUTPUT_PATH    = "./work/final_gdino_results.jsonl"
MASK_DIR       = "./work/masks"
WEIGHTS_DIR    = "./weights"

GDINO_CONFIG   = os.path.join(WEIGHTS_DIR, "GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS  = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
SAM2_WEIGHTS   = os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt")

# ── Box and mask thresholds ───────────────────────────────────────────────────
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.95
MIN_MASK_COVERAGE  = 0.001

# ── Multi-instance kinds (keep all valid boxes) ──────
MULTI_INSTANCE_KINDS = {"pest", "weed", "symptom_description", "disease"}
# Single-object kinds: keep only the highest-confidence box after QC
SINGLE_OBJECT_KINDS  = {"object", "healthy", "stress_condition"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(MASK_DIR, exist_ok=True)


# ── CDDM path fix ─────────────────────────────────────────────────────────────
def fix_image_path(path):
    if path and path.startswith(
        "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv"
    ):
        return path.replace(
            "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv",
            "./datasets/cddm/images",
            1
        )
    return path


def get_image_path(sample):
    path = sample.get("image_path")
    if not path:
        views = sample.get("views") or []
        path = views[0] if views else None
    return fix_image_path(path) if path else None


# ── Taxonomy lookup ───────────────────────────────────────────────────────────
def load_taxonomy(path):
    """Load stageB_taxonomy.json and build fast label->prompt lookup."""
    with open(path) as f:
        entries = json.load(f)
    lookup = {}
    # Priority: cddm_lookup > taxonomy_original > template
    priority = {"cddm_lookup": 0, "taxonomy_original": 1, "template": 2}
    for e in entries:
        label  = e["label"].lower().strip()
        prompt = e.get("grounding_prompt")
        source = e.get("prompt_source", "template")
        if not prompt:
            continue
        if label not in lookup:
            lookup[label] = (prompt, source)
        else:
            existing_source = lookup[label][1]
            if priority.get(source, 3) < priority.get(existing_source, 3):
                lookup[label] = (prompt, source)
    return lookup


def get_prompt(sample, taxonomy):
    """
    Get the grounding_prompt for a sample.
    Uses stageD_label to look up taxonomy. Applies synonym normalization.
    Uses stageD_crop to correct crop context in prompt.
    Handles single-word and multi-word crop names, and 'on leaf' with no crop.
    Applies prompt overrides for diseases that affect non-leaf plant parts.
    """
    import json as _json, os as _os, re as _re
    _syn_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'stageA2_synonym_map.json')
    if not _os.path.exists(_syn_path):
        raise FileNotFoundError(f'synonym map not found: {_syn_path} (it ships next to this script)')
    _synonyms = _json.load(open(_syn_path))

    label = (sample.get("stageD_label") or "").lower().strip()
    crop  = (sample.get("stageD_crop")  or "").lower().strip()

    # Prompt overrides for diseases that affect non-leaf plant parts.
    # These labels get wrong prompts from the taxonomy ("on leaf") because
    # the disease affects roots, stems, branches or bark not leaves.
    _PROMPT_OVERRIDES = {
        'root rot':               'root rot symptoms on plant',
        'black root rot':         'black root rot on plant roots',
        'crown gall':             'crown gall tumor on plant stem',
        'black knot':             'black knot fungal growth on branch',
        'fire blight':            'fire blight scorched brown branch tips',
        'botryosphaeria canker':  'botryosphaeria canker lesion on branch',
        'cytospora canker':       'cytospora canker on bark',
        'stem canker':            'stem canker lesion on stem',
        'twig blight':            'twig blight dieback on branch tips',
        'brown rot':              'brown rot on fruit',
    }

    if label in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[label], 'override'

    # Step 1 — find prompt via exact match, synonym, partial match, or fallback
    prompt, source = label, "fallback"

    if label in taxonomy:
        prompt, source = taxonomy[label]
    elif label in _synonyms:
        syn_label = _synonyms[label].lower().strip()
        if syn_label in taxonomy:
            prompt, source = taxonomy[syn_label]
    else:
        best_match = None
        best_len = 0
        for tax_label, (p, s) in taxonomy.items():
            if (len(tax_label) >= 10 and
                    tax_label in label and
                    len(tax_label) / max(len(label), 1) >= 0.5 and
                    len(tax_label) > best_len):
                best_match = (p, s)
                best_len = len(tax_label)
        if best_match:
            prompt, source = best_match

    # Step 2 — crop correction
    # Replace taxonomy default crop with actual crop from image.
    # Handles single-word crops (nectarine) and multi-word crops (lemon grass).
    # Also handles 'on leaf' with no crop word between.
    if crop and crop not in prompt.lower():
        new_prompt = _re.sub(r'on [\w\s]+ leaf', f'on {crop} leaf', prompt, count=1)
        if new_prompt == prompt:
            new_prompt = _re.sub(r'on leaf', f'on {crop} leaf', prompt, count=1)
        if new_prompt == prompt:
            new_prompt = f'{prompt} on {crop} leaf'
        prompt = new_prompt

    return prompt, source


# ── Model loading ─────────────────────────────────────────────────────────────
def load_grounding_dino():
    print("[Model] Loading GroundingDINO...")
    from groundingdino.util.inference import load_model
    model = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
    model = model.to(DEVICE)
    model.eval()
    print("[Model] GroundingDINO loaded")
    return model


def load_sam2():
    print("[Model] Loading SAM2...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=SAM2_WEIGHTS,
        device=DEVICE
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print("[Model] SAM2 loaded")
    return predictor


# ── Inference ─────────────────────────────────────────────────────────────────
def run_grounding_dino(model, image_path, text_prompt):
    from groundingdino.util.inference import load_image, predict
    try:
        image_source, image = load_image(image_path)
        h, w = image_source.shape[:2]
        boxes, logits, _ = predict(
            model=model, image=image, caption=text_prompt,
            box_threshold=0.25, text_threshold=0.20, device=DEVICE,
        )
        if len(boxes) == 0:
            return [], [], h, w

        boxes_pixel, confidences = [], []
        for box, logit in zip(boxes, logits):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = ((x2 - x1) * (y2 - y1)) / (h * w)
            if area < MIN_BOX_AREA_RATIO or area > MAX_BOX_AREA_RATIO:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_pixel.append([x1, y1, x2, y2])
            confidences.append(round(float(logit), 4))

        # Dedup: keep first occurrence of each exact box, matching confidence
        seen = set()
        deduped_boxes, deduped_conf = [], []
        for b, c in zip(boxes_pixel, confidences):
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                deduped_boxes.append(b)
                deduped_conf.append(c)
        boxes_pixel, confidences = deduped_boxes, deduped_conf

        return boxes_pixel, confidences, h, w
    except Exception as e:
        print(f"  GDino error: {e}")
        return [], [], 0, 0


def run_sam2(predictor, image_path, boxes):
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        if not boxes:
            return None
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array(boxes), multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None
        combined = (
            np.any(masks[:, 0, :, :], axis=0)
            if masks.ndim == 4
            else np.any(masks, axis=0)
        )
        coverage = combined.sum() / combined.size
        if coverage < MIN_MASK_COVERAGE or coverage > MAX_BOX_AREA_RATIO:
            return None
        return combined.astype(np.uint8) * 255
    except Exception as e:
        print(f"  SAM2 error: {e}")
        return None


def apply_box_keeping_rules(boxes, confidences, kind):
    """
    - Multi-instance targets (disease, pest, weed, symptom): keep ALL valid boxes
    - Single-object targets (object, healthy, stress): keep only best box
    """
    if not boxes:
        return boxes, confidences
    if kind in SINGLE_OBJECT_KINDS and len(boxes) > 1:
        best_idx = confidences.index(max(confidences))
        return [boxes[best_idx]], [confidences[best_idx]]
    return boxes, confidences


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"Stage E — GroundingDINO+SAM2 | Run {run_id}")
    print(f"{'='*60}\n")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load taxonomy
    print(f"\n[Taxonomy] Loading {TAXONOMY_PATH}")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"[Taxonomy] {len(taxonomy)} groundable labels ready")

    # Load selected samples
    print(f"\n[Samples] Loading {SAMPLES_PATH}")
    samples = []
    with open(SAMPLES_PATH) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                continue
    print(f"[Samples] {len(samples)} samples loaded")

    # Load models
    gdino_model    = load_grounding_dino()
    sam2_predictor = load_sam2()

    # Reset GPU memory stats for accurate measurement
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Results tracking
    counts = {
        "total":           len(samples),
        "skipped_no_image": 0,
        "skipped_no_prompt": 0,
        "skipped_no_boxes": 0,
        "boxes_generated":  0,
        "masks_generated":  0,
    }

    print(f"\n[Pipeline] Processing {len(samples)} samples...\n")

    # Resume: load already completed IDs
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        print(f"[Resume] Loading completed IDs...")
        with open(OUTPUT_PATH) as _f:
            for _line in _f:
                try:
                    completed_ids.add(json.loads(_line)["sample_id"])
                except:
                    continue
        print(f"[Resume] {len(completed_ids)} already done, skipping")

    with open(OUTPUT_PATH, "a") as fout:
        for sample in tqdm(samples, desc="GDino+SAM2"):
            # Resume — skip already processed samples
            if sample.get("sample_id") in completed_ids:
                continue
            result = sample.copy()
            kind   = sample.get("stageD_kind", "disease")

            # Get image path
            image_path = get_image_path(sample)
            if not image_path or not os.path.exists(image_path):
                counts["skipped_no_image"] += 1
                result.update({
                    "stageE_model":          "gdino+sam2",
                    "stageE_prompt":         None,
                    "stageE_prompt_source":  None,
                    "stageE_boxes":          [],
                    "stageE_confidences":    [],
                    "stageE_mask_path":      None,
                    "stageE_runtime_seconds": 0,
                    "stageE_gpu_memory_mb":  0,
                    "stageE_hit":            False,
                    "stageE_skipped_reason": "no_image",
                })
                fout.write(json.dumps(result) + "\n")
                continue

            # Get grounding prompt from taxonomy
            prompt, prompt_source = get_prompt(sample, taxonomy)
            if not prompt:
                counts["skipped_no_prompt"] += 1
                result.update({
                    "stageE_model":          "gdino+sam2",
                    "stageE_prompt":         None,
                    "stageE_prompt_source":  None,
                    "stageE_boxes":          [],
                    "stageE_confidences":    [],
                    "stageE_mask_path":      None,
                    "stageE_runtime_seconds": 0,
                    "stageE_gpu_memory_mb":  0,
                    "stageE_hit":            False,
                    "stageE_skipped_reason": "no_prompt",
                })
                fout.write(json.dumps(result) + "\n")
                continue

            # Run GroundingDINO + SAM2 and time it
            t_start = time.time()

            boxes, confidences, h, w = run_grounding_dino(
                gdino_model, image_path, prompt
            )

            # Apply box keeping rules
            boxes, confidences = apply_box_keeping_rules(
                boxes, confidences, kind
            )

            mask_path = None
            if boxes:
                counts["boxes_generated"] += 1
                mask_array = run_sam2(sam2_predictor, image_path, boxes)
                if mask_array is not None:
                    mask_path = os.path.join(
                        MASK_DIR,
                        f"{sample.get('sample_id','unknown')}_gdino_mask.png"
                    )
                    cv2.imwrite(mask_path, mask_array)
                    counts["masks_generated"] += 1
            else:
                counts["skipped_no_boxes"] += 1

            t_end = time.time()
            runtime = round(t_end - t_start, 3)

            # GPU memory
            gpu_mb = 0
            if DEVICE == "cuda":
                gpu_mb = round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 1
                )

            result.update({
                "stageE_model":           "gdino+sam2",
                "stageE_prompt":          prompt,
                "stageE_prompt_source":   prompt_source,
                "stageE_boxes":           boxes,
                "stageE_confidences":     confidences,
                "stageE_mask_path":       mask_path,
                "stageE_runtime_seconds": runtime,
                "stageE_gpu_memory_mb":   gpu_mb,
                "stageE_hit":             len(boxes) > 0,
                "stageE_skipped_reason":  None,
            })
            fout.write(json.dumps(result) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS — GDino+SAM2 | Run {run_id}")
    print(f"{'='*60}")
    for k, v in counts.items():
        print(f"  {k:<25}: {v}")
    hit_rate = counts["boxes_generated"] / max(counts["total"], 1) * 100
    print(f"  {'hit_rate':<25}: {hit_rate:.1f}%")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print(f"{'='*60}\n")

    # Save summary JSON
    summary = {
        "run_id":      run_id,
        "model":       "gdino+sam2",
        "counts":      counts,
        "hit_rate":    round(hit_rate, 2),
        "output_path": OUTPUT_PATH,
    }
    summary_path = OUTPUT_PATH.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()