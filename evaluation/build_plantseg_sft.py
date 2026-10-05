"""
Builds the expert-label comparison data from PlantSeg (nothing from it is redistributed).

Reads a local copy of PlantSeg (CC BY-NC-ND 4.0; research use only), converts each
expert mask into boxes by taking its connected components (components smaller than
0.1% of the image are dropped), and writes the result in the same grounding-only
instruction format as AgroGround. Any PlantSeg image at perceptual-hash distance 6
or less from a benchmark image is dropped first. The output is the training file of
the expert-label baseline (9,120 examples, three epochs).

The script also writes the AgroGround comparison subsets used for the scaling study:
an equal-size random subset of 9,120 examples (seed 11), and 10K and 100K subsets.

phash_index.json and the training split in the work directory.
Output: plantseg_sft_train.jsonl and the subset files in the work directory. These
derived files are not part of the release; this script lets anyone rebuild them from
their own copy of PlantSeg.

Input: the PlantSeg release (images, masks and Metadata.csv) under ./datasets/plantseg,
phash_index.json, human_verified_ground_truth_v4.jsonl and agroground_train.jsonl in the
work directory. Output: plantseg_sft_train.jsonl and the subset files in the work directory.

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""
import json, os, csv, random, numpy as np
from collections import Counter
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
import imagehash
random.seed(11)
PS="./datasets/plantseg"; OUT="./work"
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None
def u64(hs): return np.uint64(int(hs,16))
def popcnt(x): return np.unpackbits(x.view(np.uint8).reshape(-1,8),axis=1).sum(axis=1)

def main():
    # gold hashes (v4)
    idx=json.load(open(f"{OUT}/phash_index.json"))['gold']; gh=[]
    for l in open(f"{OUT}/human_verified_ground_truth_v4.jsonl"):
        r=json.loads(l); hs=r.get('phash') or idx.get(fix(r['orig_path']))
        if hs: gh.append(u64(hs))
    goldarr=np.array(gh,dtype=np.uint64)
    meta=[r for r in csv.DictReader(open(f"{PS}/Metadata.csv"))]
    train=[r for r in meta if r["Training/Test"].strip().lower().startswith("train")]
    print(f"PlantSeg metadata rows {len(meta)}; training rows {len(train)}", flush=True)
    paths=[f"{PS}/images/train/{r['Name']}" for r in train]
    with Pool(os.cpu_count()) as pool: ph=dict(pool.map(h, paths, chunksize=64))
    leak=[p for p,hs in ph.items() if hs and int(popcnt(goldarr ^ u64(hs)).min())<=6]
    print(f"PlantSeg train images near a gold image (<=6): {len(leak)} -> dropped", flush=True)
    leakset=set(leak); rows=[]; kinds=Counter(); nbox=[]
    for r in train:
        p=f"{PS}/images/train/{r['Name']}"
        if p in leakset or not ph.get(p): continue
        disease=" ".join(r["Disease"].strip().lower().split()); m=np.array(Image.open(f"{PS}/annotations/train/{r['Label file']}"))>0
        if m.sum()==0: continue
        H,W=m.shape; lab,n=ndimage.label(m); objs=ndimage.find_objects(lab)
        boxes=[]
        for sl in objs:
            y1,y2=sl[0].start,sl[0].stop; x1,x2=sl[1].start,sl[1].stop
            if (y2-y1)*(x2-x1) < 0.001*H*W: continue
            boxes.append((x1,y1,x2,y2,(y2-y1)*(x2-x1)))
        if not boxes: continue
        boxes.sort(key=lambda b:-b[4]); big=boxes[0][4] > 0.5*H*W
        def nb(b): return [int(round(b[0]/W*1000)),int(round(b[1]/H*1000)),int(round(b[2]/W*1000)),int(round(b[3]/H*1000))]
        if big or len(boxes)==1 and boxes[0][4] > 0.5*H*W:
            ys,xs=np.where(m); bb=[int(xs.min()/W*1000),int(ys.min()/H*1000),int(np.ceil(xs.max()/W*1000)),int(np.ceil(ys.max()/H*1000))]
            human=f"<image>\nLocate the plant affected by {disease} on plant. Output its bounding box in JSON format."
            gpt=json.dumps([{"bbox_2d":bb,"label":disease}]); g="coarse_object"; k=1
        else:
            bs=[nb(b) for b in boxes[:30]]
            human=f"<image>\nLocate the {disease} on plant in this image. Output the bounding boxes of the affected regions in JSON format."
            gpt=json.dumps([{"bbox_2d":b,"label":disease} for b in bs]); g="fine_region"; k=len(bs)
        kinds[g]+=1; nbox.append(k)
        rows.append({"image":p,"conversations":[{"from":"human","value":human},{"from":"gpt","value":gpt}],"granularity_level":g,"dataset":"plantseg","source_sample_id":r["Name"]})
    random.shuffle(rows)
    with open(f"{OUT}/plantseg_sft_train.jsonl","w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"PLANTSEG SFT: {len(rows)} examples | {dict(kinds)} | mean boxes {np.mean(nbox):.2f} | diseases {len(set(json.loads(r['conversations'][1]['value'])[0]['label'] for r in rows))}", flush=True)
    # equal-size pseudo subset + scaling subsets from agroground_train.jsonl (grounding-only data)
    lines=open(f"{OUT}/agroground_train.jsonl").read().splitlines(); random.shuffle(lines)
    for name,n in (("agroground_pseudo_eq",len(rows)),("agroground_sub10k",10000),("agroground_sub100k",100000)):
        open(f"{OUT}/{name}.jsonl","w").write("\n".join(lines[:n])+"\n"); print(f"{name}: {n} examples")


if __name__ == "__main__":
    main()
