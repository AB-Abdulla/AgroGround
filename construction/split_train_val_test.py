"""
80/10/10 train/val/test split of the Qwen-format examples (construction step 6).

Input: agroground_qwen_all.jsonl (from convert_to_qwen.py).

Examples are grouped by source_sample_id, so the lesion-level and object-level
examples made from the same sample always land in the same split, and the split
is stratified by dataset, so every one of the eight sources keeps the same 80/10/10
proportions. Note that the grouping is by sample, not by photograph: the sources
reuse images across samples, so the same image can occur in more than one split.
For this reason all evaluation in the paper uses the separate human-verified
benchmark, which is disjoint from the training split by file path and by
perceptual hash.

Output (in the work directory):
  agroground_train.jsonl, agroground_val.jsonl, agroground_test.jsonl
  split_summary.json with the counts per split and per dataset

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
"""
import json, random
from collections import defaultdict, Counter

INPUT = "./work/agroground_qwen_all.jsonl"
TRAIN = "./work/agroground_train.jsonl"
VAL   = "./work/agroground_val.jsonl"
TEST  = "./work/agroground_test.jsonl"
SUMMARY = "./work/split_summary.json"

TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  # test = remaining 0.10
random.seed(42)

print("Loading converted examples...")
examples = []
with open(INPUT) as f:
    for line in f:
        try:
            examples.append(json.loads(line))
        except:
            continue
print(f"Total examples: {len(examples):,}")

# Group examples by source_sample_id, tracking each group's dataset
# (all examples of one source image share the same dataset)
groups = defaultdict(list)          # source_id -> list of example indices
group_dataset = {}                  # source_id -> dataset
for i, ex in enumerate(examples):
    sid = ex.get("source_sample_id")
    groups[sid].append(i)
    group_dataset[sid] = ex.get("dataset", "unknown")

print(f"Unique source images (groups): {len(groups):,}")

# Organize group-ids by dataset for stratified assignment
groups_by_dataset = defaultdict(list)
for sid, ds in group_dataset.items():
    groups_by_dataset[ds].append(sid)

# Assign whole groups to splits, stratified per dataset
split_of_group = {}   # source_id -> 'train'/'val'/'test'
for ds, sids in groups_by_dataset.items():
    sids = sorted(sids)          # deterministic
    random.shuffle(sids)         # seeded shuffle
    n = len(sids)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    for j, sid in enumerate(sids):
        if j < n_train:
            split_of_group[sid] = "train"
        elif j < n_train + n_val:
            split_of_group[sid] = "val"
        else:
            split_of_group[sid] = "test"

# Write out — keep ONLY the fields Qwen needs (image, conversations)
# plus we keep granularity/dataset in a side file? No — Qwen ignores extra keys,
# but to be safe and clean we write the full record (Qwen reads image+conversations).
def clean_record(ex):
    # Qwen training needs 'image' and 'conversations'. Keep extra meta harmlessly.
    return ex

counts = defaultdict(lambda: Counter())   # split -> dataset -> count
gran_counts = defaultdict(lambda: Counter())  # split -> granularity -> count
img_counts = defaultdict(set)             # split -> set of source ids

fouts = {"train": open(TRAIN,"w"), "val": open(VAL,"w"), "test": open(TEST,"w")}
for ex in examples:
    sid = ex.get("source_sample_id")
    sp = split_of_group[sid]
    fouts[sp].write(json.dumps(clean_record(ex)) + "\n")
    counts[sp][ex.get("dataset","?")] += 1
    gran_counts[sp][ex.get("granularity_level","?")] += 1
    img_counts[sp].add(sid)
for f in fouts.values():
    f.close()

# Verify NO leakage: no source_id in more than one split
train_ids = img_counts["train"]; val_ids = img_counts["val"]; test_ids = img_counts["test"]
leak_tv = train_ids & val_ids
leak_tt = train_ids & test_ids
leak_vt = val_ids & test_ids

print("\n" + "="*60)
print("SPLIT COMPLETE")
print("="*60)
for sp in ["train","val","test"]:
    total = sum(counts[sp].values())
    print(f"\n{sp.upper()}: {total:,} examples from {len(img_counts[sp]):,} images")
    print(f"  granularity: fine={gran_counts[sp].get('fine_region',0):,}  coarse={gran_counts[sp].get('coarse_object',0):,}")
    print(f"  by dataset:")
    for ds in sorted(counts[sp].keys()):
        print(f"    {ds:<12}: {counts[sp][ds]:,}")

print("\n" + "="*60)
print("LEAKAGE CHECK (must all be 0):")
print(f"  train∩val : {len(leak_tv)}")
print(f"  train∩test: {len(leak_tt)}")
print(f"  val∩test  : {len(leak_vt)}")
print("="*60)

summary = {
    "total_examples": len(examples),
    "unique_images": len(groups),
    "train": {"examples": sum(counts['train'].values()), "images": len(img_counts['train']),
              "by_dataset": dict(counts['train']), "by_granularity": dict(gran_counts['train'])},
    "val":   {"examples": sum(counts['val'].values()), "images": len(img_counts['val']),
              "by_dataset": dict(counts['val']), "by_granularity": dict(gran_counts['val'])},
    "test":  {"examples": sum(counts['test'].values()), "images": len(img_counts['test']),
              "by_dataset": dict(counts['test']), "by_granularity": dict(gran_counts['test'])},
    "leakage": {"train_val": len(leak_tv), "train_test": len(leak_tt), "val_test": len(leak_vt)},
    "method": "group split by source_sample_id, stratified by dataset, 80/10/10, seed=42",
}
with open(SUMMARY,"w") as f:
    json.dump(summary, f, indent=2)
print(f"\nOutputs: agroground_train/val/test.jsonl")
print(f"Summary: {SUMMARY}")
