"""
Pixel-level verification of the near-duplicate pairs flagged by neardup_check.py.

For every flagged benchmark/training pair, both images are loaded, resized to a
common size, and compared by pixel correlation. Pairs with correlation above 0.97
are verified copies, pairs between 0.90 and 0.97 are ambiguous (excluded from the
benchmark conservatively), and pairs below 0.90 are hash collisions that are kept.
The paper's audit found 564 verified copies, 8 ambiguous pairs and 25 collisions.

Input: the flagged pairs from neardup_check.py and the image files. Output:
gold_neardup_verified.json with the verdict and correlation for every pair.
"""


import json, numpy as np
from PIL import Image
from collections import Counter, defaultdict
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
gold={json.loads(l)['coco_id']:json.loads(l) for l in open('./work/human_verified_ground_truth_v3.jsonl')}
hits=json.load(open('./work/gold_neardup_hits.json'))
def thumb(p): return np.asarray(Image.open(p).convert("L").resize((96,96), Image.BILINEAR), dtype=np.float32)
res=[]
for h in hits:
    a=thumb(fix(gold[h['coco_id']]['orig_path'])); b=thumb(h['train_match'])
    corr=float(np.corrcoef(a.ravel(), b.ravel())[0,1]); mad=float(np.abs(a-b).mean())
    res.append({**h, "corr":corr, "mad":mad, "same_size": Image.open(fix(gold[h['coco_id']]['orig_path'])).size==Image.open(h['train_match']).size})
json.dump(res, open('./work/gold_neardup_verified.json','w'))
true=[r for r in res if r['corr']>0.97]; ambiguous=[r for r in res if 0.90<r['corr']<=0.97]; false=[r for r in res if r['corr']<=0.90]
print(f"hits {len(res)}: TRUE duplicates (pixel corr>0.97): {len(true)} | ambiguous (0.90-0.97): {len(ambiguous)} | hash collisions (<=0.90): {len(false)}")
print("true dups by dataset   :", dict(Counter(r['dataset'] for r in true)))
print("true dups by granularity:", dict(Counter(r['granularity'] for r in true)))
print("true dups same pixel size:", sum(r['same_size'] for r in true), "/", len(true))
print("corr by phash distance :", {d: f"{np.mean([r['corr'] for r in res if r['dist']==d]):.3f}" for d in sorted(set(r['dist'] for r in res))})
idx=json.load(open('./work/phash_index.json'))
print(f"\nTRAINING: {len(idx['train']):,} unique file paths -> {len(set(v for v in idx['train'].values() if v)):,} unique perceptual hashes")
gh=[v for v in idx['gold'].values() if v]; print(f"GOLD: {len(gh)} images -> {len(set(gh))} unique hashes (within-gold duplicates: {len(gh)-len(set(gh))})")
