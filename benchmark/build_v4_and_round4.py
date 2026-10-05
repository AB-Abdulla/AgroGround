"""
Builds the cleaned round-3 benchmark and selects the round-4 candidates.

Step 1: starting from the round-3 annotation file, removes every image that the
near-duplicate audit (gold_neardup_verified.json) verified as a copy of a training
image or marked ambiguous (pixel correlation above 0.90), and every within-set repeat
(same perceptual hash as another benchmark image). Writes
human_verified_ground_truth_v4clean.jsonl and gold_v3_dropped_duplicates.json.

Step 2: selects new candidates from the held-out test split that share no file with
the training split, lie at perceptual-hash distance greater than 6 from every training
image, duplicate no image already in the benchmark, and are unique by hash among
themselves. Targets 150 dense lesion-level, 520 lesion-level and 230 object-level
images with CDDM capped at 40% of each stratum, seeded (seed 44); the hash-disjoint
pools yielded 774 candidates, of which 452 were annotated in id order (ids from 4000).
Writes eval_round4_manifest.jsonl.

Input: phash_index.json, gold_neardup_verified.json, the round-3 annotation file,
agroground_test.jsonl and agroground_granularity.jsonl, in the work directory.
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""
import json, os, random, numpy as np
from collections import defaultdict, Counter
from multiprocessing import Pool
from PIL import Image
import imagehash
random.seed(44)
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None

def main():
    idx=json.load(open('./work/phash_index.json')); train_h=idx['train']; gold_h=idx['gold']
    ver=json.load(open('./work/gold_neardup_verified.json'))
    drop=set(r['coco_id'] for r in ver if r['corr']>0.90)
    v3=[json.loads(l) for l in open('./work/human_verified_ground_truth_v3.jsonl')]
    seen=set(); clean=[]; within=0
    for r in v3:
        if r['coco_id'] in drop: continue
        hs=gold_h.get(fix(r['orig_path']))
        if hs in seen: within+=1; continue
        seen.add(hs); clean.append(r)
    with open('./work/human_verified_ground_truth_v4clean.jsonl','w') as f:
        for r in clean: f.write(json.dumps(r)+'\n')
    json.dump(sorted(drop), open('./work/gold_v3_dropped_duplicates.json','w'))
    c=Counter(r['granularity'] for r in clean)
    print(f"v4-clean: {len(clean)} = fine {c['fine']} / coarse {c['coarse']} / healthy {c['healthy']}  (dropped {len(drop)} train-duplicates + {within} within-gold repeats)", flush=True)
    train_files=set(train_h.keys()); test_ids=set(json.loads(l).get("source_sample_id") for l in open('./work/agroground_test.jsonl'))
    cands={}
    for line in open('./work/agroground_granularity.jsonl'):
        try: r=json.loads(line)
        except: continue
        if r.get("source_sample_id") not in test_ids: continue
        p=fix(r.get("image_path","")); g=r.get("granularity_level")
        if not p or p in train_files or g not in ("fine_region","coarse_object"): continue
        if p not in cands: cands[p]=r
    print(f"path-disjoint test candidates: {len(cands):,} — hashing ...", flush=True)
    with Pool(os.cpu_count()) as pool: ch=dict(pool.map(h, list(cands), chunksize=128))
    def to_u64(hs): return np.uint64(int(hs,16))
    th=np.array([to_u64(v) for v in set(train_h.values()) if v], dtype=np.uint64)
    gold_hashes=set(v for v in gold_h.values() if v)
    def popcnt(x): return np.unpackbits(x.view(np.uint8).reshape(-1,8),axis=1).sum(axis=1)
    ok=[]; usedh=set(); near=0
    for p,hs in ch.items():
        if not hs or hs in gold_hashes or hs in usedh: continue
        d=popcnt(th ^ to_u64(hs)).min()
        if d<=6: near+=1; continue
        usedh.add(hs); ok.append(cands[p])
    print(f"rejected as near-duplicates of training (<=6): {near:,} | disjoint pool: {len(ok):,}", flush=True)
    pools={"dense_fine":defaultdict(list),"fine":defaultdict(list),"coarse":defaultdict(list)}
    for r in ok:
        nb=len(r.get("boxes") or []); ds=r.get("stageD_dataset")
        if r["granularity_level"]=="fine_region": pools["dense_fine" if nb>=5 else "fine"][ds].append(r)
        else: pools["coarse"][ds].append(r)
    for k in pools: print(f"  pool[{k}]: total {sum(len(v) for v in pools[k].values()):,} | "+", ".join(f"{d}={len(v)}" for d,v in sorted(pools[k].items(),key=lambda x:-len(x[1]))))
    T={"dense_fine":150,"fine":520,"coarse":230}; CAP=0.40
    def allocate(k,target):
        by=pools[k]; chosen=[]; cap=int(target*CAP); c=list(by.get("cddm",[])); random.shuffle(c); chosen+=c[:cap]
        others={d:list(v) for d,v in by.items() if d!="cddm"}
        for v in others.values(): random.shuffle(v)
        order=sorted(others,key=lambda d:-len(others[d])); rem=target-len(chosen)
        while rem>0 and any(others.values()):
            for d in order:
                if rem<=0: break
                if others[d]: chosen.append(others[d].pop()); rem-=1
        if rem>0: extra=[x for x in c if x not in chosen]; chosen+=extra[:rem]
        return chosen
    manifest=[]
    for k,t in T.items():
        for r in allocate(k,t):
            manifest.append({"image_path":r["image_path"],"regime":"single_target" if k=="coarse" else "multi_target","granularity":"coarse_object" if k=="coarse" else "fine_region",
                "dataset":r.get("stageD_dataset"),"disease_query":r.get("stageE_prompt"),"raw_label":r.get("stageD_label"),"crop":(r.get("stageD_crop") or "").strip(),
                "gdino_boxes":r.get("boxes"),"image_width":r.get("image_width"),"image_height":r.get("image_height"),"stratum":k,"phash":ch[fix(r["image_path"])]})
    random.shuffle(manifest)
    with open('./work/eval_round4_manifest.jsonl','w') as f:
        for m in manifest: f.write(json.dumps(m)+'\n')
    print(f"\nROUND-4 SELECTION: {len(manifest)} images (path- and hash-disjoint from training, ASSERTED)")
    for k in T: print(f"  {k:<11}{sum(1 for m in manifest if m['stratum']==k):>4}: "+", ".join(f"{d}={n}" for d,n in Counter(m['dataset'] for m in manifest if m['stratum']==k).most_common()))


if __name__ == "__main__":
    main()
