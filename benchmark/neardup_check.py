"""
Perceptual-hash index and near-duplicate check between the benchmark and the training data.

Computes a 64-bit perceptual hash (pHash, 8x8 DCT) for every unique training image
file and every benchmark image, writes them to phash_index.json, and reports every
benchmark/training pair at Hamming distance 6 or less as a near-duplicate candidate.
The flagged pairs are then verified at pixel level by neardup_verify.py.

Input: the benchmark annotation file and the training split in the work directory,
and the image files. Output: phash_index.json and the list of flagged pairs with their
distances.
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""
import json, os, sys
from multiprocessing import Pool
from PIL import Image
import imagehash
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None
if __name__=="__main__":
    gold=[json.loads(l) for l in open('./work/human_verified_ground_truth_v3.jsonl')]
    train=sorted(set(fix(json.loads(l)['image']) for l in open('./work/agroground_train.jsonl')))
    print(f"hashing {len(train):,} training files + {len(gold)} gold images ...", flush=True)
    with Pool(os.cpu_count()) as pool:
        th=dict(pool.map(h, train, chunksize=256)); gh=dict(pool.map(h, [fix(g['orig_path']) for g in gold], chunksize=16))
    json.dump({"train":th,"gold":gh}, open('./work/phash_index.json','w'))
    from collections import defaultdict, Counter
    bucket=defaultdict(list)
    for p,hs in th.items():
        if hs: bucket[hs].append(p)
    def ham(a,b): return bin(int(a,16)^int(b,16)).count("1")
    thr=int(sys.argv[1]) if len(sys.argv)>1 else 6
    train_hashes=[(imagehash.hex_to_hash(k),k) for k in bucket]
    hits=[]
    for g in gold:
        hs=gh.get(fix(g['orig_path']))
        if not hs: continue
        gv=imagehash.hex_to_hash(hs)
        best=min(((gv-tv),k) for tv,k in train_hashes)
        if best[0]<=thr: hits.append((g['coco_id'],g['dataset'],g['granularity'],best[0],bucket[best[1]][0]))
    print(f"\nNEAR-DUPLICATES (pHash Hamming <= {thr} of 64 bits): {len(hits)} / {len(gold)} gold images")
    print("  by dataset   :", dict(Counter(x[1] for x in hits))); print("  by granularity:", dict(Counter(x[2] for x in hits)))
    print("  distance hist:", dict(sorted(Counter(x[3] for x in hits).items())))
    json.dump([{"coco_id":c,"dataset":d,"granularity":g,"dist":dist,"train_match":m} for c,d,g,dist,m in hits], open('./work/gold_neardup_hits.json','w'))
    for c,d,g,dist,m in hits[:6]: print(f"   gold {c} [{d}/{g}] dist {dist} ~ {m.split('/')[-1]}")
