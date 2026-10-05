"""
Label extraction and groundability routing for AgroGround (construction step 1).

Reads every sample of the eight source datasets from their unified JSONL files,
runs each one through the handler written for that dataset (stageA_label_counts.py),
and keeps the samples whose answer names a visually localizable target. For each
kept sample the handler gives the target label, its kind (disease, pest, weed,
symptom description, object or stress condition) and the host crop, and the
grounding prompt is looked up in stageB_taxonomy.json.

Samples whose answer is a management action or a cause are routed out, and so are
healthy samples (SKIP_HEALTHY = True); they are written separately by
capture_nongroundable.py so that the two partitions sum to the ingested total.
The run is resumable and the CDDM image paths are mapped to the local copy.

Input: unified_jsonl/*.jsonl (one file per dataset) and stageB_taxonomy.json
Output: final_samples.jsonl, one record per groundable sample, with the original
sample fields plus:
    stageD_kind: target kind
    stageD_label:  normalized label from the handler
    stageD_crop: host crop, when available
    stageD_dataset: dataset name
    stageD_prompt: grounding prompt from the taxonomy
    stageD_prompt_source: taxonomy_original, cddm_lookup, template or fallback
and final_preprocess_summary.json with counts per dataset and per kind.

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

# ── Add working directory to path so we can import stageA_label_counts ────────
sys.path.insert(0, '.')
from stageA_label_counts import (
    handle_agrobench_sample,
    handle_agrocot_sample,
    handle_cddm_sample,
    handle_agmmu_sample,
    handle_agromind_sample,
    handle_leafbench_sample,
    handle_leafnet_sample,
    handle_mirage_sample,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
JSONL_DIR      = "./work/unified_jsonl"
TAXONOMY_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stageB_taxonomy.json")
OUTPUT_PATH    = "./work/final_samples.jsonl"
SUMMARY_PATH   = "./work/final_preprocess_summary.json"

# ── Dataset handlers (same order as Stage D) ──────────────────────────────────
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

# ── Kinds routed out of the groundable set ─────────────────────────────────────
# action and cause are not groundable; healthy samples are routed to the
# non-groundable stream (SKIP_HEALTHY = True), where capture_nongroundable.py keeps them


SKIP_HEALTHY   = True
NOT_GROUNDABLE = {"action", "cause"}


# For agromind
# ── Universal label blocklist (aerial field imagery artifacts) ─────────────────
LABEL_BLOCKLIST = {
    # Aerial field imagery artifacts
    'weed_cluster', 'double_plant', 'cloud_shadow',
    'standing_water', 'waterway', 'planter_skip',
    # Non-agricultural / too generic
    'raccoon', 'ball python', 'fungus', 'plant',
    # Bare crop names — produce wrong prompts like "apple on leaf"
    # (plant species ID questions, not disease/pest grounding)
    'apple', 'black pepper', 'cashew', 'cassava', 'cherry',
    'coffee', 'cucumber', 'grape', 'maize', 'mango', 'orange',
    'peach', 'pepper', 'potato', 'raspberry', 'rice', 'soybean',
    'squash', 'strawberry', 'sugarcane', 'tea', 'tomato',
}





# ── CDDM path fix ─────────────────────────────────────────────────────────────
def fix_image_path(path):
    if path and path.startswith(
        "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv"
    ):
        return path.replace(
            "/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv",
            "./datasets/cddm/images",
            1,
        )
    return path

def get_image_path(sample):
    path = sample.get("image_path")
    if not path:
        views = sample.get("views") or []
        path = views[0] if views else None
    return fix_image_path(path) if path else None

# ── Taxonomy lookup (the same logic as in final_gdino_script.py) ─────────────
def load_taxonomy(path):
    with open(path) as f:
        entries = json.load(f)
    lookup = {}
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

def get_prompt(label, taxonomy, crop=None):
    """
    Get the grounding_prompt for a label string.
    Applies synonym normalization from stageA2_synonym_map.json.
    Uses crop to correct crop context in prompt.
    Applies prompt overrides for diseases that affect non-leaf plant parts.
    """
    import json as _json, os as _os, re as _re
    _syn_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'stageA2_synonym_map.json')
    if not _os.path.exists(_syn_path):
        raise FileNotFoundError(f'synonym map not found: {_syn_path} (it ships next to this script)')
    _synonyms = _json.load(open(_syn_path))

    label_lower = label.lower().strip()
    crop_lower  = (crop or "").lower().strip()

    # Prompt overrides for diseases that affect non-leaf plant parts
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

    if label_lower in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[label_lower], 'override'

    # Step 1 — find prompt via exact match, synonym, partial match, or fallback
    prompt, source = label_lower, "fallback"

    if label_lower in taxonomy:
        prompt, source = taxonomy[label_lower]
    elif label_lower in _synonyms:
        syn_label = _synonyms[label_lower].lower().strip()
        if syn_label in taxonomy:
            prompt, source = taxonomy[syn_label]
    else:
        best_match = None
        best_len = 0
        for tax_label, (p, s) in taxonomy.items():
            if (len(tax_label) >= 10 and
                    tax_label in label_lower and
                    len(tax_label) / max(len(label_lower), 1) >= 0.5 and
                    len(tax_label) > best_len):
                best_match = (p, s)
                best_len = len(tax_label)
        if best_match:
            prompt, source = best_match

    # Step 2 — crop correction
    if crop_lower and crop_lower not in prompt.lower():
        new_prompt = _re.sub(r'on [\w\s]+ leaf', f'on {crop_lower} leaf', prompt, count=1)
        if new_prompt == prompt:
            new_prompt = _re.sub(r'on leaf', f'on {crop_lower} leaf', prompt, count=1)
        if new_prompt == prompt:
            new_prompt = f'{prompt} on {crop_lower} leaf'
        prompt = new_prompt

    return prompt, source


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"Final Preprocess | Run {run_id}")
    print(f"{'='*60}\n")

    # Load taxonomy
    print(f"[Taxonomy] Loading {TAXONOMY_PATH}")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"[Taxonomy] {len(taxonomy)} groundable labels ready\n")

    # Check for existing output (resume capability)
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        print(f"[Resume] Found existing output. Loading completed sample IDs...")
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r["sample_id"])
                except:
                    continue
        print(f"[Resume] {len(completed_ids)} samples already processed. Skipping.\n")

    # Counters
    counts = {
        "total_read":        0,
        "handler_none":      0,
        "not_groundable":    0,
        "healthy_skipped":   0,
        "no_image":          0,
        "already_done":      0,
        "written":           0,
    }
    per_dataset  = defaultdict(Counter)   # dataset -> kind -> count
    per_kind     = Counter()              # kind -> count

    t_start = time.time()

    with open(OUTPUT_PATH, "a") as fout:
        for ds_name, handler in DATASET_HANDLERS.items():
            jsonl_path = os.path.join(JSONL_DIR, f"{ds_name}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"[{ds_name}] File not found, skipping")
                continue

            ds_written = 0
            ds_skipped = 0

            with open(jsonl_path) as f:
                for line in tqdm(f, desc=f"{ds_name:<12}", unit=" samples"):
                    counts["total_read"] += 1
                    try:
                        sample = json.loads(line)
                    except:
                        continue

                    sample_id = sample.get("sample_id", "")

                    # Resume: skip already processed
                    if sample_id in completed_ids:
                        counts["already_done"] += 1
                        continue

                    # Run handler
                    result = handler(sample)
                    if result is None:
                        counts["handler_none"] += 1
                        continue

                    kind  = result.get("kind", "")
                    label = result.get("label", "")
                    crop  = result.get("crop")



                    # Skip not groundable kinds (action, cause)
                    if kind in NOT_GROUNDABLE:
                        counts["not_groundable"] += 1
                        continue

                    # Skip CDDM object samples — plant ID sentences not groundable
                    if ds_name == 'cddm' and kind == 'object':
                        counts["not_groundable"] += 1
                        continue

                    # Skip aerial field imagery artifacts across all datasets
                    if label.lower().strip() in LABEL_BLOCKLIST:
                        counts["not_groundable"] += 1
                        continue


                    # Skip healthy if configured
                    if SKIP_HEALTHY and kind == "healthy":
                        counts["healthy_skipped"] += 1
                        continue

                    # Check image exists
                    image_path = get_image_path(sample)
                    if not image_path or not os.path.exists(image_path):
                        counts["no_image"] += 1
                        continue

                    # Get grounding prompt
                    prompt, prompt_source = get_prompt(label, taxonomy, crop=crop)

                    # Build the output record
                    record = {
                        "sample_id":            sample_id,
                        "source":               ds_name,
                        "image_path":           image_path,
                        "views":                sample.get("views"),
                        "question":             sample.get("question", ""),
                        "options":              sample.get("options", {}),
                        "answer":               sample.get("answer", ""),
                        "answer_letter":        sample.get("answer_letter"),
                        "metadata":             sample.get("metadata", {}),
                        "boxes":                sample.get("boxes"),
                        "masks":                sample.get("masks"),
                        "stageD_kind":          kind,
                        "stageD_label":         label,
                        "stageD_crop":          crop,
                        "stageD_dataset":       ds_name,
                        "stageD_prompt":        prompt,
                        "stageD_prompt_source": prompt_source,
                    }

                    fout.write(json.dumps(record) + "\n")
                    counts["written"] += 1
                    per_dataset[ds_name][kind] += 1
                    per_kind[kind] += 1
                    ds_written += 1

            print(f"  {ds_name}: {ds_written} written")

    # Summary
    elapsed = round(time.time() - t_start, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS — Final Preprocess | Run {run_id}")
    print(f"{'='*60}")
    print(f"\nRuntime: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"\nCounts:")
    for k, v in counts.items():
        print(f"  {k:<25}: {v:>10,}")

    print(f"\nPer kind:")
    for kind, count in sorted(per_kind.items()):
        print(f"  {kind:<25}: {count:>10,}")

    print(f"\nPer dataset:")
    for ds_name in DATASET_HANDLERS:
        if ds_name in per_dataset:
            total = sum(per_dataset[ds_name].values())
            parts = ", ".join(f"{k}:{v}" for k, v in sorted(per_dataset[ds_name].items()))
            print(f"  {ds_name:<12}: {total:>8,}  ({parts})")

    print(f"\nOutput: {OUTPUT_PATH}")

    # Save summary
    summary = {
        "run_id":       run_id,
        "runtime_s":    elapsed,
        "counts":       counts,
        "per_kind":     dict(per_kind),
        "per_dataset":  {ds: dict(k) for ds, k in per_dataset.items()},
        "output_path":  OUTPUT_PATH,
        "skip_healthy": SKIP_HEALTHY,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {SUMMARY_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()