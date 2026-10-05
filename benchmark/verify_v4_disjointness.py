"""
Final disjointness check of the released benchmark against the training data.

This is the audit script named in the paper's reproducibility statement. For the
1,480 images of human_verified_ground_truth_v4.jsonl it verifies:
  (i)   no benchmark image file occurs in the training split (positives) or among the
        healthy images used as training negatives;
  (ii)  every training image, positive or negative, at perceptual-hash distance 6 or less
        (64-bit pHash, 8x8 DCT) from a benchmark image is a hash collision, not a copy:
        every flagged pair (all training hashes within distance 6 of a benchmark image, not
        only the nearest) is verified at pixel level exactly as in neardup_verify.py
        (grayscale, 96x96 bilinear, Pearson correlation); correlation above 0.97 is a
        verified copy and 0.90 to 0.97 is ambiguous, both of which fail the check, while
        0.90 or below is a collision, which is reported and allowed (the paper kept 25);
  (iii) no two benchmark images lie at distance 6 or less from each other.
It then prints the composition (lesion-level, object-level, healthy; per source; box
counts) and exits with status 1 if any check fails.

Hashes are read from phash_index.json (training positives and benchmark images, as
written by neardup_check.py); the check fails if any training-split file lacks a hash; hashes of the negatives are computed here and cached in
phash_negatives.json. Images that are missing locally are reported and counted as
unverified rather than silently passed.

Input (work directory): human_verified_ground_truth_v4.jsonl, agroground_train.jsonl,
agroground_recogneg2_train.jsonl, phash_index.json, and the image files.
Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""
import os
import sys
import json
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image
import imagehash

WORK = "./work"
GOLD = f"{WORK}/human_verified_ground_truth_v4.jsonl"
TRAIN = f"{WORK}/agroground_train.jsonl"
NEG = f"{WORK}/agroground_recogneg2_train.jsonl"
INDEX = f"{WORK}/phash_index.json"
NEG_CACHE = f"{WORK}/phash_negatives.json"
MAX_DIST = 6


def fix(p):
    return p.replace("/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv", "./datasets/cddm/images") if p and "/home/jovyan" in p else p


def phash(p):
    try:
        with Image.open(p) as im:
            return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception:
        return p, None


def u64(hs):
    return np.uint64(int(hs, 16))


def popcnt(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def min_dist(hs, arr):
    return int(popcnt(arr ^ u64(hs)).min()) if len(arr) else 99


def thumb(p):
    return np.asarray(Image.open(p).convert("L").resize((96, 96), Image.BILINEAR), dtype=np.float32)


def pixel_corr(a, b):
    """The verification statistic of neardup_verify.py."""
    try:
        return float(np.corrcoef(thumb(a).ravel(), thumb(b).ravel())[0, 1])
    except Exception:
        return float("nan")


def neighbours(hs, by_hash, keys, arr, max_dist):
    """Every indexed image within max_dist of the hash, nearest first."""
    d = popcnt(arr ^ u64(hs))
    idx = np.where(d <= max_dist)[0]
    return [(int(d[j]), by_hash[keys[j]]) for j in sorted(idx, key=lambda j: d[j])]


def main():
    gold = [json.loads(l) for l in open(GOLD)]
    gold_paths = [fix(g["orig_path"]) for g in gold]
    train_files = set(fix(json.loads(l)["image"]) for l in open(TRAIN))
    neg_files = set()
    for l in open(NEG):
        ex = json.loads(l)
        if ex.get("task_format", "").endswith("_negative"):
            neg_files.add(fix(ex["image"]))
    print(f"benchmark images {len(gold):,} | training files {len(train_files):,} | negative files {len(neg_files):,}")

    # (i) file-path disjointness
    path_hits = [p for p in gold_paths if p in train_files or p in neg_files]
    print(f"(i)   shared file paths with training positives or negatives: {len(path_hits)}")

    # hashes
    idx = json.load(open(INDEX))
    train_h = {fix(k): v for k, v in idx["train"].items()}
    gold_h = {fix(k): v for k, v in idx["gold"].items()}
    missing_gold = [p for p in gold_paths if p not in gold_h or not gold_h[p]]
    if missing_gold:
        with Pool(os.cpu_count()) as pool:
            for p, hs in pool.map(phash, missing_gold, chunksize=16):
                gold_h[p] = hs
    neg_h = {}
    if os.path.exists(NEG_CACHE):
        neg_h = {fix(k): v for k, v in json.load(open(NEG_CACHE)).items()}
    todo = sorted(p for p in neg_files if p not in neg_h)
    if todo:
        print(f"hashing {len(todo):,} negative images ...", flush=True)
        with Pool(os.cpu_count()) as pool:
            for p, hs in pool.map(phash, todo, chunksize=256):
                neg_h[p] = hs
        json.dump(neg_h, open(NEG_CACHE, "w"))
    unverified = [p for p in gold_paths if not gold_h.get(p)]
    print(f"      benchmark images without a hash (missing image file): {len(unverified)}")
    # completeness of the training index: every file of the training split must carry a hash
    unhashed_train = [p for p in train_files if not train_h.get(p)]
    print(f"      training files without a hash in phash_index.json: {len(unhashed_train)}")

    # (ii) flagged pairs at hash distance <= 6, then pixel verification of each flagged pair
    def index(H):
        by_hash = {}
        for p, h in H.items():
            if h:
                by_hash.setdefault(h, p)
        keys = list(by_hash)
        return by_hash, keys, np.array([u64(h) for h in keys], dtype=np.uint64)
    copies, ambiguous, collisions = [], [], []
    for name, H in (("positive", train_h), ("negative", neg_h)):
        by_hash, keys, arr = index(H)
        if not len(arr):
            continue
        for g, p in zip(gold, gold_paths):
            hs = gold_h.get(p)
            if not hs:
                continue
            for d, q in neighbours(hs, by_hash, keys, arr, MAX_DIST):
                c = pixel_corr(p, q)
                rec = (g["coco_id"], g["granularity"], g["dataset"], name, d, round(c, 3), q)
                (copies if c > 0.97 else ambiguous if c > 0.90 else collisions).append(rec)
    print(f"(ii)  flagged pairs at distance <= {MAX_DIST}: {len(copies) + len(ambiguous) + len(collisions)} "
          f"-> verified copies {len(copies)}, ambiguous {len(ambiguous)}, hash collisions (kept) {len(collisions)}")
    for rec in copies + ambiguous:
        print("      FAIL", *rec)
    for rec in collisions:
        print("      collision", *rec)

    # (iii) within-benchmark near-repeats
    hashes = [(g["coco_id"], u64(gold_h[p])) for g, p in zip(gold, gold_paths) if gold_h.get(p)]
    arr = np.array([h for _, h in hashes], dtype=np.uint64)
    within = 0
    for k, (cid, h) in enumerate(hashes):
        d = popcnt(arr ^ h)
        d[k] = 99
        if d.min() <= MAX_DIST:
            within += 1
    print(f"(iii) benchmark images within distance {MAX_DIST} of another benchmark image: {within}")

    # composition
    gran = Counter(g["granularity"] for g in gold)
    src = Counter(g["dataset"] for g in gold)
    boxes = sum(len(g["gt_boxes"]) for g in gold)
    fine_boxes = sum(len(g["gt_boxes"]) for g in gold if g["granularity"] == "fine")
    print(f"composition: lesion-level {gran.get('fine', 0)}, object-level {gran.get('coarse', 0)}, healthy {gran.get('healthy', 0)} | "
          f"boxes {boxes:,} ({fine_boxes:,} lesion-level) | per source: " + ", ".join(f"{k} {v}" for k, v in src.most_common()))

    ok = not path_hits and not copies and not ambiguous and within == 0 and not unverified and not unhashed_train
    print("RESULT:", "PASS, the benchmark is disjoint from the training data" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
