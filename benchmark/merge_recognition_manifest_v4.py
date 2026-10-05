"""
Assembles the released candidate manifest, recognition_eval_manifest_v4.jsonl.

The benchmark was built in four annotation rounds. Candidate lists for the images
of rounds 1 to 3 were produced by build_recognition_eval_v3.py from the round-3
annotations. When round 4 added new images, the same builder was run again on the
new rows only (build_recognition_eval_new.py, which is the round-3 builder with its
input set to gt_new_rows.jsonl and its output to recognition_eval_manifest_new.jsonl).
This script does the two remaining steps:

  1. writes gt_new_rows.jsonl, the round-4 rows of the final annotation file
     (coco_id >= 4000), which is the input of build_recognition_eval_new.py;
  2. after that builder has run, keeps the rows of the round-3 manifest whose images
     survived into the final benchmark, appends the new rows, checks that the ids
     match the final annotation file exactly and that every candidate is lowercase,
     and writes recognition_eval_manifest_v4.jsonl.

Usage, from the work directory:
    python merge_recognition_manifest_v4.py extract      # step 1
    python build_recognition_eval_new.py                 # candidates for the new rows
    python merge_recognition_manifest_v4.py merge        # step 2

Paths: set AGROGROUND_WORK_DIR for the work directory (default ./work).
"""
import os
import sys
import json

WORK_DIR = os.environ.get("AGROGROUND_WORK_DIR", "./work")
GOLD = os.path.join(WORK_DIR, "human_verified_ground_truth_v4.jsonl")
NEW_ROWS = os.path.join(WORK_DIR, "gt_new_rows.jsonl")
OLD_MANIFEST = os.path.join(WORK_DIR, "recognition_eval_manifest_v3.jsonl")
NEW_MANIFEST = os.path.join(WORK_DIR, "recognition_eval_manifest_new.jsonl")
OUTPUT = os.path.join(WORK_DIR, "recognition_eval_manifest_v4.jsonl")


def extract():
    """Step 1: the round-4 annotation rows (coco_id >= 4000)."""
    v4 = [json.loads(l) for l in open(GOLD)]
    new = [r for r in v4 if r["coco_id"] >= 4000]
    with open(NEW_ROWS, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in new))
    print("new rows:", len(new))


def merge():
    """Step 2: surviving round-3 rows plus the new rows, checked against the gold set."""
    ids = {json.loads(l)["coco_id"] for l in open(GOLD)}
    old = [json.loads(l) for l in open(OLD_MANIFEST)]
    keep = [r for r in old if r["coco_id"] in ids]
    new = [json.loads(l) for l in open(NEW_MANIFEST)]
    rows = keep + new
    assert {r["coco_id"] for r in rows} == ids, (len(rows), len(ids))
    up = sum(any(c != c.lower() for c in r["candidates"]) or r["correct_answer"] != r["correct_answer"].lower()
             for r in rows)
    assert up == 0, f"{up} rows carry uppercase candidates or answers"
    with open(OUTPUT, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"manifest v4: {len(rows)} rows (old {len(keep)} + new {len(new)}) | uppercase rows: {up} | "
          f"healthy-none: {sum(r['correct_answer'] == 'none' for r in rows)}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "extract":
        extract()
    elif step == "merge":
        merge()
    else:
        print("usage: python merge_recognition_manifest_v4.py [extract|merge]")
