"""
Converts the training data to the mixed instruction format.

Input: agroground_train/val/test.jsonl (from split_train_val_test.py) and the
granularity file, which supplies each sample's crop and target kind.

Half of the examples are rewritten in the recognition-and-localization format:
the instruction lists four candidate targets (the true one and three distractors
matched by crop and kind, the same rule as the benchmark's candidate lists) at a
random position, without naming the answer, and asks the model to identify which
one is present and localize it. The other half keep the known-target format
unchanged. Answers are never changed and split membership is never changed.
Distractor and position choices are seeded (seed 31).

Output: agroground_recog_train/val/test.jsonl in the work directory.
"""
import json, random, re
from collections import defaultdict, Counter

random.seed(31)
GRAN = "./work/agroground_granularity.jsonl"
SPLITS = ["train", "val", "test"]
RECOG_FRACTION = 0.5
N_CANDIDATES = 4

def norm(s): return (s or "").strip().lower()
def too_similar(a, b):
    na, nb = norm(a), norm(b)
    return na == nb or na in nb or nb in na

# Pass 1: label pools + per-source crop/kind/label from granularity file 
print("Building pools from granularity file...")
pool_crop_kind = defaultdict(set); pool_ds_kind = defaultdict(set); pool_kind = defaultdict(set)
src_info = {}   # source_sample_id to (label, crop, kind, dataset)
with open(GRAN) as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        label = norm(r.get('stageD_label')) or None; kind = r.get('stageD_kind')
        crop = norm(r.get('stageD_crop')); ds = r.get('stageD_dataset')
        sid = r.get('source_sample_id')
        if sid and sid not in src_info and label:
            src_info[sid] = (label, crop, kind, ds)
        if label and kind:
            if crop: pool_crop_kind[(crop, kind)].add(label)
            if ds:   pool_ds_kind[(ds, kind)].add(label)
            pool_kind[kind].add(label)
print(f"  sources: {len(src_info):,} | crop-kind pools: {len(pool_crop_kind)}")

def pick_distractors(true_label, crop, kind, dataset, n):
    chosen = []; tried = set()
    for pool in (pool_crop_kind.get((crop, kind), set()),
                 pool_ds_kind.get((dataset, kind), set()),
                 pool_kind.get(kind, set()),
                 pool_kind.get('disease', set())):
        cands = [c for c in pool if c not in tried and not too_similar(c, true_label)]
        random.shuffle(cands)
        for c in cands:
            if any(too_similar(c, x) for x in chosen): continue
            chosen.append(c); tried.add(c)
            if len(chosen) >= n: return chosen
    return chosen

def get_true_label(gpt_value):
    """Parse the label from the answer JSON."""
    try:
        data = json.loads(gpt_value)
        for item in data:
            if isinstance(item, dict) and item.get('label'):
                return norm(str(item['label']))
    except: pass
    return None

def build_recog_instruction(candidates, granularity):
    listing = ", ".join(candidates)
    if granularity == 'coarse_object':
        ground = ("locate the affected leaf or plant. "
                  "Output the bounding box in JSON format with the identified name as the label.")
    else:
        ground = ("locate all affected regions. "
                  "Output bounding boxes in JSON format with the identified name as the label.")
    return (f"<image>\nThis plant may be affected by one of the following: {listing}. "
            f"Identify which one is present in the image and {ground} "
            f"If none of them are present, output an empty list.")

# Pass 2: convert each split
for split in SPLITS:
    inp = f"./work/agroground_{split}.jsonl"
    outp = f"./work/agroground_recog_{split}.jsonl"
    stats = Counter()
    with open(inp) as fi, open(outp, "w") as fo:
        for line in fi:
            try: ex = json.loads(line)
            except: continue
            convert = random.random() < RECOG_FRACTION
            if convert:
                gpt_val = ex['conversations'][1]['value']
                true_label = get_true_label(gpt_val)
                sid = ex.get('source_sample_id')
                info = src_info.get(sid)
                if true_label and info:
                    _, crop, kind, ds = info
                    distr = pick_distractors(true_label, crop, kind or 'disease',
                                             ds or ex.get('dataset'), N_CANDIDATES - 1)
                    if len(distr) >= 2:   # need a real choice, else keep original
                        candidates = [true_label] + distr
                        random.shuffle(candidates)   # true answer at RANDOM position
                        ex['conversations'][0]['value'] = build_recog_instruction(
                            candidates, ex.get('granularity_level', 'fine_region'))
                        ex['task_format'] = 'recognition'
                        stats['recognition'] += 1
                    else:
                        ex['task_format'] = 'grounding_given'
                        stats['kept_no_distractors'] += 1
                else:
                    ex['task_format'] = 'grounding_given'
                    stats['kept_no_label_or_info'] += 1
            else:
                ex['task_format'] = 'grounding_given'
                stats['grounding_given'] += 1
            fo.write(json.dumps(ex) + "\n")
    total = sum(stats.values())
    print(f"\n{split}: {total:,} examples -> {outp}")
    for k, v in stats.most_common():
        print(f"    {k}: {v:,} ({v/total*100:.1f}%)")

# Show 2 converted samples for verification 
print("\nSAMPLE CONVERTED EXAMPLES (verify format):")
shown = 0
with open("./work/agroground_recog_train.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        if ex.get('task_format') == 'recognition':
            print(f"\n[{ex.get('granularity_level')}] {ex.get('dataset')}")
            print("  HUMAN:", ex['conversations'][0]['value'][:260].replace('\n', ' '))
            print("  GPT  :", ex['conversations'][1]['value'][:140])
            shown += 1
            if shown >= 2: break
