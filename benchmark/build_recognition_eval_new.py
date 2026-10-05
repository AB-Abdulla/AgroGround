"""
Builds the candidate lists for the images added in annotation round 4.

This is build_recognition_eval_v3.py with its input set to gt_new_rows.jsonl (the
round-4 annotation rows, written by merge_recognition_manifest_v4.py extract) and
its output set to recognition_eval_manifest_new.jsonl. The rules are the same:
four candidates per image, the true label plus three distractors drawn from the
same crop and the same kind, falling back to the same source dataset and then to
the global pool of that kind; a candidate is rejected if, after normalization, it
equals another or is a substring of another; candidates are shuffled with a fixed
seed (seed 2026); healthy images get four plausible candidates of the same crop
and the correct answer "none". Each record also carries the recognition prompt
for the image's granularity.

merge_recognition_manifest_v4.py merge joins this output with the round-3
manifest into the released recognition_eval_manifest_v4.jsonl.

Input: gt_new_rows.jsonl and final_gdino_results.jsonl (for crops and kinds).
Output: recognition_eval_manifest_new.jsonl, in the work directory.
"""
import json, random
from collections import defaultdict

random.seed(2026)

GT_FILE  = "./work/gt_new_rows.jsonl"
GDINO_ALL = "./work/final_gdino_results.jsonl"
OUTPUT   = "./work/recognition_eval_manifest_new.jsonl"
N_CANDIDATES = 4

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

def norm(s):
    return (s or "").strip().lower()

# ---- Pass over full annotation data: label pools + per-image crop/kind ----
print("Building label pools from full dataset (one pass)...")
pool_crop_kind = defaultdict(set)     # (crop, kind) -> labels
pool_ds_kind   = defaultdict(set)     # (dataset, kind) -> labels
pool_kind      = defaultdict(set)     # kind -> labels
path_info      = {}                   # fixed path -> (crop, kind)

with open(GDINO_ALL) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        label = norm(r.get('stageD_label')) or None; kind = r.get('stageD_kind')
        crop = norm(r.get('stageD_crop')); ds = r.get('stageD_dataset')
        p = fix_path(r.get('image_path'))
        if p and p not in path_info:
            path_info[p] = (crop, kind)
        if label and kind:
            if crop: pool_crop_kind[(crop, kind)].add(label)
            if ds:   pool_ds_kind[(ds, kind)].add(label)
            pool_kind[kind].add(label)

print(f"  crop-kind pools: {len(pool_crop_kind)}, dataset-kind pools: {len(pool_ds_kind)}")

def too_similar(a, b):
    """Guard: don't use a distractor that's essentially the same label."""
    na, nb = norm(a), norm(b)
    return na == nb or na in nb or nb in na

def pick_distractors(true_label, crop, kind, dataset, n):
    chosen = []
    tried = set()
    for pool in (pool_crop_kind.get((crop, kind), set()),
                 pool_ds_kind.get((dataset, kind), set()),
                 pool_kind.get(kind, set()),
                 pool_kind.get('disease', set())):
        cands = [c for c in pool
                 if c not in tried and not too_similar(c, true_label)]
        random.shuffle(cands)
        for c in cands:
            if any(too_similar(c, x) for x in chosen):
                continue
            chosen.append(c); tried.add(c)
            if len(chosen) >= n: return chosen
    return chosen  # may be < n for very sparse pools

def build_prompt(candidates, granularity):
    listing = ", ".join(candidates)
    if granularity == 'coarse':
        ground = ("locate the affected leaf or plant. "
                  "Output the bounding box in JSON format with the identified name as the label.")
    else:  # fine + healthy use fine-style
        ground = ("locate all affected regions. "
                  "Output bounding boxes in JSON format with the identified name as the label.")
    return (f"This plant may be affected by one of the following: {listing}. "
            f"Identify which one is present in the image and {ground} "
            f"If none of them are present, output an empty list.")

# ---- Build manifest ----
records = []
stats = {'crop_matched': 0, 'fallback': 0, 'short_list': 0}
with open(GT_FILE) as f:
    gts = [json.loads(l) for l in f]

for g in gts:
    if g['granularity'] not in ('fine','coarse','healthy'):
        continue
    p = g['orig_path']
    crop, kind = path_info.get(p, ('', 'disease'))
    if g['granularity'] == 'healthy':
        crop = norm(g.get('crop')) or crop
        kind = 'disease'
        true_label = None
        correct = "none"
        n_needed = N_CANDIDATES
        base = norm(g.get('abstention_disease')) or None
        # healthy: candidates = abstention disease + 3 more distractors
        distractors = pick_distractors(base or "", crop, kind, g['dataset'], n_needed-1)
        candidates = ([base] if base else []) + distractors
    else:
        true_label = norm(g.get('disease_raw')) or 'unknown'
        correct = true_label
        distractors = pick_distractors(true_label, crop, kind, g['dataset'], N_CANDIDATES-1)
        candidates = [true_label] + distractors
        if pool_crop_kind.get((crop, kind)) and len(distractors) == N_CANDIDATES-1:
            stats['crop_matched'] += 1
        else:
            stats['fallback'] += 1
    if len(candidates) < N_CANDIDATES:
        stats['short_list'] += 1
    random.shuffle(candidates)
    records.append({
        'coco_id': g['coco_id'],
        'orig_path': p,
        'dataset': g['dataset'],
        'granularity': g['granularity'],
        'crop': crop, 'kind': kind,
        'candidates': candidates,
        'correct_answer': correct,          # "none" for healthy
        'gt_boxes': g['gt_boxes'],
        'image_width': g['image_width'], 'image_height': g['image_height'],
        'prompt': build_prompt(candidates, g['granularity']),
    })

with open(OUTPUT, 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

# ---- Summary + samples ----
from collections import Counter
print(f"\n{'='*70}\nRECOGNITION EVAL MANIFEST BUILT\n{'='*70}")
print(f"Total: {len(records)}  |  by granularity: {dict(Counter(r['granularity'] for r in records))}")
print(f"Crop-matched candidate lists: {stats['crop_matched']}, fallback-supplemented: {stats['fallback']}, short lists: {stats['short_list']}")
print(f"\nSAMPLES (verify candidates look sensible):")
shown = {'fine':0,'coarse':0,'healthy':0}
for r in records:
    if shown[r['granularity']] >= 2: continue
    shown[r['granularity']] += 1
    print(f"\n[{r['granularity'].upper()}] {r['dataset']} | crop={r['crop'] or '?'} | kind={r['kind']}")
    print(f"  candidates: {r['candidates']}")
    print(f"  correct   : {r['correct_answer']}")
print(f"\nSaved: {OUTPUT}")
