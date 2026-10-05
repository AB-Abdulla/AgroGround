"""
Round 4b: healthy images for the target-absent regime.

Selects healthy images that share no file with, and lie at perceptual-hash distance
greater than 6 from, every training image (positives and the healthy images used as
negatives) and every image already in the benchmark. Crop metadata is required: each
image is assigned a crop-matched disease query (a random disease of the same crop),
which is the target the model must report as absent. Selection is seeded (seed 45).

Input: phash_index.json, agroground_recogneg2_train.jsonl (the negatives),
final_gdino_results.jsonl (crop metadata) and final_nongroundable_samples.jsonl, in the
work directory. Output: eval_round4b_manifest.jsonl.
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""
import json, os, random, numpy as np
from collections import defaultdict, Counter
from multiprocessing import Pool
from PIL import Image
import imagehash
random.seed(45)
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def norm(s): return (s or "").strip().lower()
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None
def u64(hs): return np.uint64(int(hs,16))
def popcnt(x): return np.unpackbits(x.view(np.uint8).reshape(-1,8),axis=1).sum(axis=1)
def mind(hs, arr): return int(popcnt(arr ^ u64(hs)).min()) if len(arr) else 99

def main():
    idx=json.load(open('./work/phash_index.json')); train_files=set(idx['train'].keys())
    trainarr=np.array([u64(v) for v in set(idx['train'].values()) if v],dtype=np.uint64)
    neg=set()
    for line in open('./work/agroground_recogneg2_train.jsonl'):
        ex=json.loads(line)
        if ex.get("task_format","").endswith("_negative"): neg.add(fix(ex["image"]))
    gold_paths=set(); gold_hashes=[]
    for f in ('human_verified_ground_truth_v3.jsonl','human_verified_ground_truth_v4clean.jsonl'):
        for l in open('./work/'+f):
            r=json.loads(l); p=fix(r['orig_path']); gold_paths.add(p)
            hs=idx['gold'].get(p)
            if hs: gold_hashes.append(u64(hs))
    goldarr=np.array(gold_hashes,dtype=np.uint64)
    pools=defaultdict(set)
    for line in open('./work/final_gdino_results.jsonl'):
        try: r=json.loads(line)
        except: continue
        if r.get('stageD_label') and r.get('stageD_kind')=='disease' and r.get('stageD_crop'): pools[norm(r['stageD_crop'])].add(norm(r['stageD_label']))
    cands={}
    for line in open('./work/final_nongroundable_samples.jsonl'):
        try: r=json.loads(line)
        except: continue
        if r.get('nongroundable_reason')!='healthy': continue
        p=fix(r.get('image_path','')); crop=norm(r.get('stageD_crop') or r.get('crop'))
        if not p or not crop or p in train_files or p in neg or p in gold_paths or crop not in pools: continue
        if p not in cands: cands[p]=(r, crop)
    print(f"healthy candidates (path-disjoint, crop known): {len(cands):,} — hashing ...", flush=True)
    with Pool(os.cpu_count()) as pool: ch=dict(pool.map(h, list(cands), chunksize=128))
    print("hashing training negatives ...", flush=True)
    with Pool(os.cpu_count()) as pool: nh=dict(pool.map(h, sorted(neg), chunksize=256))
    negarr=np.array([u64(v) for v in set(nh.values()) if v],dtype=np.uint64)
    ok=[]; sel=[]; rej=Counter()
    for p,(r,crop) in cands.items():
        hs=ch.get(p)
        if not hs: continue
        if mind(hs,trainarr)<=6: rej['train positive']+=1; continue
        if mind(hs,negarr)<=6: rej['train negative']+=1; continue
        if mind(hs,goldarr)<=6: rej['gold']+=1; continue
        ok.append((p,r,crop,hs))
    print(f"rejected: {dict(rej)} | disjoint healthy pool: {len(ok):,} | by dataset: {dict(Counter(r.get('stageD_dataset') for _,r,_,_ in ok).most_common())}", flush=True)
    random.shuffle(ok); by=defaultdict(list)
    for x in ok: by[x[1].get('stageD_dataset')].append(x)
    TARGET=320; cap=int(TARGET*0.40); chosen=by.get('cddm',[])[:cap]; others={d:v for d,v in by.items() if d!='cddm'}
    order=sorted(others,key=lambda d:-len(others[d])); rem=TARGET-len(chosen)
    while rem>0 and any(others.values()):
        for d in order:
            if rem<=0: break
            if others[d]: chosen.append(others[d].pop()); rem-=1
    selh=[]; final=[]
    for p,r,crop,hs in chosen:
        if selh and mind(hs,np.array(selh,dtype=np.uint64))<=6: continue
        selh.append(u64(hs))
        with Image.open(p) as im: W,H=im.size
        final.append({"image_path":p,"regime":"target_absent","granularity":"healthy","dataset":r.get('stageD_dataset'),"crop":crop,
                      "abstention_disease":random.choice(sorted(pools[crop])),"gdino_boxes":None,"raw_label":"healthy","disease_query":"healthy plant (confirm no disease; delete if diseased)",
                      "image_width":W,"image_height":H,"stratum":"healthy","phash":hs})
    with open('./work/eval_round4b_manifest.jsonl','w') as f:
        for m in final: f.write(json.dumps(m)+'\n')
    print(f"ROUND-4b HEALTHY SELECTION: {len(final)} images (disjoint from all training positives, negatives, and gold — ASSERTED) | datasets: {dict(Counter(m['dataset'] for m in final).most_common())}")


if __name__ == "__main__":
    main()
