"""
Pre-annotation exclusion filter for the cleaned benchmark and the round-4 candidates.

Run after build_v4_and_round4.py and before round 4 was annotated. It hashes the
healthy images used as training negatives (from agroground_recogneg2_train.jsonl) and
applies two exclusions to the cleaned round-3 benchmark (human_verified_ground_truth_v4clean.jsonl):
healthy benchmark images at perceptual-hash distance 6 or less from a training
negative, and within-benchmark near-repeats. It then filters the round-4 candidate
manifest the same way (near an existing benchmark image, near a healthy training
image, or a within-round repeat). The excluded ids and paths are written to
gold_v4clean_excluded_ids.json and round4_excluded_paths.json and were applied when the
final benchmark was assembled.

This script is not the final disjointness check of the released benchmark; that is
verify_v4_disjointness.py.

Input: agroground_recogneg2_train.jsonl, phash_index.json,
human_verified_ground_truth_v4clean.jsonl and eval_round4_manifest.jsonl, in the work
directory.
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""

import json, os, numpy as np
from collections import Counter
from multiprocessing import Pool
from PIL import Image
import imagehash
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None
def u64(hs): return np.uint64(int(hs,16))
def popcnt(x): return np.unpackbits(x.view(np.uint8).reshape(-1,8),axis=1).sum(axis=1)
def mind(hs, arr): return int(popcnt(arr ^ u64(hs)).min()) if len(arr) else 99

def main():
    neg=set()
    for line in open('./work/agroground_recogneg2_train.jsonl'):
        ex=json.loads(line)
        if ex.get("task_format","").endswith("_negative"): neg.add(fix(ex["image"]))
    print(f"healthy negative training images: {len(neg):,} — hashing ...", flush=True)
    with Pool(os.cpu_count()) as pool: nh=dict(pool.map(h, sorted(neg), chunksize=256))
    negarr=np.array([u64(v) for v in set(nh.values()) if v], dtype=np.uint64)
    gold_h=json.load(open('./work/phash_index.json'))['gold']
    clean=[json.loads(l) for l in open('./work/human_verified_ground_truth_v4clean.jsonl')]
    excl_gold=[]; kept=[]; reason=Counter()
    for r in clean:
        hs=gold_h.get(fix(r['orig_path']))
        if hs and r['granularity']=='healthy' and mind(hs,negarr)<=6: excl_gold.append(r['coco_id']); reason['healthy~training negative']+=1; continue
        if hs and kept and mind(hs,np.array(kept,dtype=np.uint64))<=6: excl_gold.append(r['coco_id']); reason['within-gold near-repeat']+=1; continue
        if hs: kept.append(u64(hs))
    man=[json.loads(l) for l in open('./work/eval_round4_manifest.jsonl')]
    goldarr=np.array(kept,dtype=np.uint64); excl_r4=[]; sel=[]; r4=Counter()
    for m in man:
        hs=m['phash']
        if mind(hs,goldarr)<=6: excl_r4.append(m['image_path']); r4['near existing gold']+=1; continue
        if mind(hs,negarr)<=6: excl_r4.append(m['image_path']); r4['near healthy training image']+=1; continue
        if sel and mind(hs,np.array(sel,dtype=np.uint64))<=6: excl_r4.append(m['image_path']); r4['within-round repeat']+=1; continue
        sel.append(u64(hs))
    json.dump(excl_gold, open('./work/gold_v4clean_excluded_ids.json','w')); json.dump(excl_r4, open('./work/round4_excluded_paths.json','w'))
    print(f"v4-clean: {len(clean)-len(excl_gold)} kept, {len(excl_gold)} excluded {dict(reason)}")
    print(f"round-4: {len(man)-len(excl_r4)} valid of {len(man)} uploaded, {len(excl_r4)} excluded {dict(r4)}")
    print("EXCLUSION LISTS SAVED - annotate everything; the merge applies them")


if __name__ == "__main__":
    main()
