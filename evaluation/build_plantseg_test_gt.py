"""
Builds the PlantSeg test-set ground truth for the external transfer evaluation.

Reads a local copy of PlantSeg (CC BY-NC-ND 4.0; research use only) and converts
the expert masks of its test split into boxes by connected components (components
smaller than 0.1% of the image are dropped). Images with an empty mask or with no
component above the threshold are skipped. Images that are perceptual-hash
near-duplicates of AgroGround training files are removed (119 in the paper's run),
leaving 2,165 test images. Masks are matched to images by file name, because the
release's metadata points every test row at one mask file.

Granularity is assigned from the components (lesion-level or object-level) so that
the same metrics as on the benchmark can be reported.

Input: the PlantSeg release (images, masks and Metadata.csv) under ./datasets/plantseg
and phash_index.json in the work directory.
Output: plantseg_test_gt.jsonl in the work directory. This derived file is not
part of the release.

Paths are relative to the repository root: intermediate files under ./work, source
datasets under ./datasets (one sub-folder per source).
Uses multiprocessing with the fork start method; run it as a script on Linux.
"""

import json, csv, os, numpy as np
from collections import Counter
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
import imagehash
PS="./datasets/plantseg"
def h(p):
    try:
        with Image.open(p) as im: return p, str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception: return p, None
def u64(hs): return np.uint64(int(hs,16))
def popcnt(x): return np.unpackbits(x.view(np.uint8).reshape(-1,8),axis=1).sum(axis=1)

def main():
    idx=json.load(open('./work/phash_index.json'))
    trainarr=np.array([u64(v) for v in set(idx['train'].values()) if v],dtype=np.uint64)
    meta=[r for r in csv.DictReader(open(f"{PS}/Metadata.csv")) if r["Training/Test"].strip().lower().startswith("test")]
    paths=[f"{PS}/images/test/{r['Name']}" for r in meta]
    with Pool(os.cpu_count()) as pool: ph=dict(pool.map(h, paths, chunksize=64))
    rows=[]; near=0; missing=0; kinds=Counter(); cid=90000
    for r in meta:
        p=f"{PS}/images/test/{r['Name']}"; hs=ph.get(p)
        if not hs or not os.path.exists(f"{PS}/annotations/test/{os.path.splitext(r['Name'].strip())[0]}.png"): missing+=1; continue
        if int(popcnt(trainarr ^ u64(hs)).min())<=6: near+=1; continue
        disease=" ".join(r["Disease"].strip().lower().split()); m=np.array(Image.open(f"{PS}/annotations/test/{os.path.splitext(r['Name'].strip())[0]}.png"))>0
        if m.sum()==0: continue
        H,W=m.shape; lab,n=ndimage.label(m); boxes=[]
        for sl in ndimage.find_objects(lab):
            y1,y2=sl[0].start,sl[0].stop; x1,x2=sl[1].start,sl[1].stop
            if (y2-y1)*(x2-x1) >= 0.001*H*W: boxes.append([float(x1),float(y1),float(x2),float(y2)])
        if not boxes: continue
        big=max((b[2]-b[0])*(b[3]-b[1]) for b in boxes) > 0.5*H*W
        if big:
            ys,xs=np.where(m); boxes=[[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]]; g="coarse"
        else: g="fine"; boxes=sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))[:30]
        kinds[g]+=1; cid+=1
        rows.append({"coco_id":cid,"orig_path":p,"dataset":"plantseg_test","granularity":g,"gt_boxes":boxes,"image_width":W,"image_height":H,"disease_raw":disease,"crop":r["Plant"].strip().lower(),"phash":hs})
    open('./work/plantseg_test_gt.jsonl','w').write(''.join(json.dumps(x)+'\n' for x in rows))
    print(f"PlantSeg test: {len(meta)} images; {missing} missing image/mask files; {near} dropped as near-duplicates of our training files; GT rows {len(rows)} {dict(kinds)}; mean boxes/fine {np.mean([len(x['gt_boxes']) for x in rows if x['granularity']=='fine']):.2f}")


if __name__ == "__main__":
    main()
