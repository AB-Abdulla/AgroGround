"""
Round 3 candidate selection for the human-verified benchmark.

Selects new positive images under the file-path rule introduced in this round: a
candidate is rejected if its image file occurs in the training split. Candidates are
drawn from the held-out test split; any image already in the benchmark or in the
round-1 and round-2 manifests is excluded; CDDM is capped at 40% of each stratum.
Selection is seeded (seed 33).

Targets: 200 dense lesion-level images (at least 5 detector boxes), 300 further
lesion-level images and 200 object-level images. A first pass with targets 200/200/160
returned 425 images because only 65 dense images were image-disjoint from training;
the targets were then raised to 200/300/200 and the run selected 565 images
(65 dense, 300 lesion-level, 200 object-level), which is the manifest that was
annotated. No healthy images are added in this round.

Input: agroground_granularity.jsonl, agroground_test.jsonl, agroground_train.jsonl,
the benchmark annotation file as it stood after round 2, and the round-1 and round-2
manifests, in the work directory. Output: eval_round3_manifest.jsonl (same schema as
the earlier rounds).
"""
import json, random
from collections import defaultdict, Counter
random.seed(33)
GRAN="./work/agroground_granularity.jsonl"; TEST="./work/agroground_test.jsonl"
TRAIN="./work/agroground_train.jsonl"; GOLD="./work/human_verified_ground_truth_v2.jsonl"
R1="./work/eval_1000_manifest.jsonl"; R2="./work/eval_round2_manifest.jsonl"
OUT="./work/eval_round3_manifest.jsonl"
T={"dense_fine":200,"fine":300,"coarse":200}; CDDM_CAP=0.40; DENSE=5
def fix(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
print("loading exclusions ...", flush=True)
train_files=set(fix(json.loads(l)["image"]) for l in open(TRAIN))
excl=set(fix(json.loads(l)["orig_path"]) for l in open(GOLD))
for f in (R1,R2): excl|=set(fix(json.loads(l)["image_path"]) for l in open(f))
print(f"  train files {len(train_files):,} | gold+prior-selected excluded {len(excl):,}", flush=True)
test_ids=set(json.loads(l).get("source_sample_id") for l in open(TEST))
pools={k:defaultdict(list) for k in T}; seen=set(); n_twin=0
for line in open(GRAN):
    try: r=json.loads(line)
    except: continue
    if r.get("source_sample_id") not in test_ids: continue
    p=fix(r.get("image_path",""))
    if not p or p in excl or p in seen: continue
    if p in train_files: n_twin+=1; continue          # <-- image-disjoint rule
    g=r.get("granularity_level"); ds=r.get("stageD_dataset"); nb=len(r.get("boxes") or [])
    if g=="fine_region": pools["dense_fine" if nb>=DENSE else "fine"][ds].append(r); seen.add(p)
    elif g=="coarse_object": pools["coarse"][ds].append(r); seen.add(p)
print(f"  test-split candidates rejected as train twins: {n_twin:,}")
for k in T: print(f"  pool[{k}] (disjoint): total {sum(len(v) for v in pools[k].values()):,} | "+", ".join(f"{d}={len(v)}" for d,v in sorted(pools[k].items(),key=lambda x:-len(x[1]))))
def allocate(k,target):
    by=pools[k]; chosen=[]; cap=int(target*CDDM_CAP)
    c=by.get("cddm",[]); random.shuffle(c); chosen+=c[:cap]
    others={d:list(v) for d,v in by.items() if d!="cddm"}
    for v in others.values(): random.shuffle(v)
    order=sorted(others,key=lambda d:-len(others[d])); rem=target-len(chosen)
    while rem>0 and any(others.values()):
        for d in order:
            if rem<=0: break
            if others[d]: chosen.append(others[d].pop()); rem-=1
    if rem>0: extra=[x for x in c if x not in chosen]; chosen+=extra[:rem]
    return chosen
manifest=[]; rep=defaultdict(Counter)
for k,t in T.items():
    for r in allocate(k,t):
        regime="single_target" if k=="coarse" else "multi_target"; gran="coarse_object" if k=="coarse" else "fine_region"
        manifest.append({"image_path":r["image_path"],"regime":regime,"granularity":gran,"dataset":r.get("stageD_dataset"),
            "disease_query":r.get("stageE_prompt"),"raw_label":r.get("stageD_label"),"crop":(r.get("stageD_crop") or "").strip(),
            "gdino_boxes":r.get("boxes"),"image_width":r.get("image_width"),"image_height":r.get("image_height"),"stratum":k})
        rep[k][r.get("stageD_dataset")]+=1
random.shuffle(manifest)
assert all(fix(m["image_path"]) not in train_files for m in manifest), "TRAIN TWIN LEAKED"
with open(OUT,"w") as f:
    for m in manifest: f.write(json.dumps(m)+"\n")
print(f"\nROUND-3 SELECTION: {len(manifest)} images (target {sum(T.values())}) — image-disjoint from training ASSERTED")
for k in T: print(f"  {k:<11}{sum(rep[k].values()):>4}: "+", ".join(f"{d}={c}" for d,c in sorted(rep[k].items(),key=lambda x:-x[1])))
