"""
Adds healthy negative examples to the mixed training data at a 12% share.

Input: the mixed-format split files (agroground_recog_train/val.jsonl, from
build_recognition_training_data.py), the granularity file and the non-groundable
file (for the healthy pool), and eval_healthy_paths_all.txt, the list of the 430 healthy images selected for the
benchmark in rounds 1 and 2 (written by benchmark/select_eval_round2.py), which are
excluded from the pool; the script asserts that the lock holds exactly 430 paths so
that it cannot run against a stale lock.

Healthy images are drawn from the pool with seed 47. Each selected image gives two
negatives: one in the candidate format (four plausible diseases of the same crop,
answer an empty list) and one in the known-target format (a plausible disease of
the same crop, answer an empty list). Half of the known-target positives (seed 51)
are rewritten with the same "if none is present, output an empty list" phrasing as
the negatives, and the negatives use both phrasings, so that no sentence pattern
predicts the empty answer. Answers are never changed and the split is untouched.

Counts: 86,000 train negatives (43,000 images) and 10,750 val negatives, 11.9% of
the resulting training mixture (721,977 examples).

Output: agroground_recogneg2_train.jsonl and agroground_recogneg2_val.jsonl in the
work directory. This is the data of the main supervised model.
"""
import json, random
from collections import defaultdict, Counter

GRAN = "./work/agroground_granularity.jsonl"
NONG = "./work/final_nongroundable_samples.jsonl"
LOCK = "./work/eval_healthy_paths_all.txt"
N_TRAIN_NEG, N_VAL_NEG = 86000, 10750

def norm(s): return (s or "").strip().lower()
def too_similar(a, b):
    na, nb = norm(a), norm(b); return na == nb or na in nb or nb in na
def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p
def get_true_label(gpt_value):
    try:
        for item in json.loads(gpt_value):
            if isinstance(item, dict) and item.get('label'): return str(item['label'])
    except: pass
    return None

CLAUSE_FINE   = ("<image>\nLocate all regions affected by {d} in this image. "
                 "Output bounding boxes in JSON format. If no such regions are present, output an empty list.")
CLAUSE_COARSE = ("<image>\nLocate the leaf or plant affected by {d} in this image. "
                 "Output the bounding box in JSON format. If not present, output an empty list.")
PLAIN_FINE    = ("<image>\nDetect the specific lesions of {d} on {c} and return their bounding boxes in JSON format.")
PLAIN_COARSE  = ("<image>\nDetect the {c} leaf or plant affected by {d} and return its bounding box in JSON format.")

print("Pass 1: label pools...")
random.seed(47)
pool_ck = defaultdict(set); pool_dk = defaultdict(set); pool_k = defaultdict(set)
with open(GRAN) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        label = norm(r.get('stageD_label')) or None; kind = r.get('stageD_kind')
        crop = norm(r.get('stageD_crop')); ds = r.get('stageD_dataset')
        if label and kind:
            if crop: pool_ck[(crop, kind)].add(label)
            if ds:   pool_dk[(ds, kind)].add(label)
            pool_k[kind].add(label)

def pick_diseases(crop, dataset, n, avoid=()):
    chosen = []
    for pool in (pool_ck.get((crop,'disease'), set()), pool_dk.get((dataset,'disease'), set()), pool_k.get('disease', set())):
        cands = list(pool); random.shuffle(cands)
        for c in cands:
            if any(too_similar(c, x) for x in list(chosen)+list(avoid)): continue
            chosen.append(c)
            if len(chosen) >= n: return chosen
    return chosen

print("Pass 2: healthy pool (seed 47)...")
locked = set(l.strip() for l in open(LOCK)); assert len(locked) == 430
healthy, seen, skl = [], set(), 0
with open(NONG) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get('nongroundable_reason') != 'healthy': continue
        crop = norm(r.get('stageD_crop'))
        if not crop: continue
        p = fix_path(r.get('image_path') or '')
        if not p or p in seen: continue
        if p in locked: skl += 1; seen.add(p); continue
        seen.add(p)
        healthy.append({"path": p, "crop": crop, "dataset": r.get('stageD_dataset') or 'unknown',
                        "sid": r.get('source_sample_id') or p})
print(f"  unique healthy: {len(healthy):,} | eval-locked excluded: {skl}")
assert skl > 0
random.shuffle(healthy)
btr, bva = int((N_TRAIN_NEG//2)*1.05), int((N_VAL_NEG//2)*1.05)
train_imgs, val_imgs = healthy[:btr], healthy[btr:btr+bva]

def cand_instruction(candidates, coarse):
    listing = ", ".join(candidates)
    ground = ("locate the affected leaf or plant. Output the bounding box in JSON format with the identified name as the label."
              if coarse else
              "locate all affected regions. Output bounding boxes in JSON format with the identified name as the label.")
    return (f"<image>\nThis plant may be affected by one of the following: {listing}. "
            f"Identify which one is present in the image and {ground} "
            f"If none of them are present, output an empty list.")

def two_negatives(h, i):
    coarse1, coarse2 = (i % 2 == 0), (i % 2 == 1)
    use_clause = (i % 4 < 2)                       # 50/50 clause vs plain for the given-format negative
    cands = pick_diseases(h["crop"], h["dataset"], 4)
    if len(cands) < 4: return None
    gd = pick_diseases(h["crop"], h["dataset"], 1, avoid=cands)
    gd = gd[0] if gd else cands[0]
    if use_clause:
        gtxt = (CLAUSE_COARSE if coarse2 else CLAUSE_FINE).format(d=gd)
    else:
        gtxt = (PLAIN_COARSE if coarse2 else PLAIN_FINE).format(d=gd, c=h["crop"])
    base = {"image": h["path"], "granularity_level": "non_groundable",
            "dataset": h["dataset"], "source_sample_id": h["sid"]}
    e1 = dict(base, task_format="recognition_negative",
              conversations=[{"from":"human","value":cand_instruction(cands, coarse1)},{"from":"gpt","value":"[]"}])
    e2 = dict(base, task_format="grounding_given_negative", phrasing=("clause" if use_clause else "plain"),
              conversations=[{"from":"human","value":gtxt},{"from":"gpt","value":"[]"}])
    return e1, e2

rng_conv = random.Random(51)   # independent stream for positive-phrasing conversion
for split, pool, target in (("train", train_imgs, N_TRAIN_NEG), ("val", val_imgs, N_VAL_NEG)):
    lines, conv, kept = [], 0, 0
    with open(f"./work/agroground_recog_{split}.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("task_format") == "grounding_given" and rng_conv.random() < 0.5:
                d = get_true_label(ex['conversations'][1]['value'])
                if d:
                    tpl = CLAUSE_COARSE if ex.get('granularity_level') == 'coarse_object' else CLAUSE_FINE
                    ex['conversations'][0]['value'] = tpl.format(d=d)
                    ex['phrasing'] = 'clause'; conv += 1
                else: kept += 1
            lines.append(json.dumps(ex))
    n_pos = len(lines)
    made, fmt = 0, Counter()
    for i, h in enumerate(pool):
        if made >= target: break
        pair = two_negatives(h, i)
        if pair is None: continue
        for e in pair:
            lines.append(json.dumps(e)); made += 1
            fmt[e["task_format"] + ("/" + e.get("phrasing","") if e.get("phrasing") else "")] += 1
    random.shuffle(lines)
    out = f"./work/agroground_recogneg2_{split}.jsonl"
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"\n{split}: {n_pos:,} positives ({conv:,} converted to clause phrasing, {kept} unparseable kept) "
          f"+ {made:,} negatives = {len(lines):,} -> {out}")
    print(f"  negative formats: {dict(fmt)}")

print("\nSAMPLES (one of each new kind):")
want = {"pos_clause": 0, "grounding_given_negative/plain": 0, "grounding_given_negative/clause": 0}
with open("./work/agroground_recogneg2_train.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        key = None
        if ex.get("task_format") == "grounding_given" and ex.get("phrasing") == "clause": key = "pos_clause"
        elif ex.get("task_format") == "grounding_given_negative":
            key = "grounding_given_negative/" + ex.get("phrasing","")
        if key in want and want[key] < 1:
            want[key] += 1
            print(f"\n[{key}]")
            print("  HUMAN:", ex['conversations'][0]['value'][:200].replace("\n"," "))
            print("  GPT  :", ex['conversations'][1]['value'][:100])
        if all(v >= 1 for v in want.values()): break
