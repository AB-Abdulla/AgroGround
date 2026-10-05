"""
Builds the rollout prompt set for the GRPO stage as verl parquet files.

Prompts are drawn (seed 2027) from the training split of the 6% negatives file
(agroground_recogneg3_train/val.jsonl; its positives are identical to the 12% file's),
rewritten with the evaluation phrasings, and oversampled toward dense images: 4,000
prompts in each of four positive cells (known-target and candidate format, each as
dense images with at least 4 weak boxes and as other images) and 1,500 in each of two
healthy cells, 19,000 requested in total. The sampled rows whose image exceeds 0.5
megapixels are then dropped (a selection, not a resize; at run time this was applied
to the written parquet files in successive passes, the last at 0.5 MP, leaving the
18,750 training prompts reported in the paper). The reward's ground truth is the
example's weak labels. A check asserts that no benchmark image is present (the released
script checks against the final benchmark file; the run-time check used the round-3
file). Also writes a small smoke-test parquet.

Output: train_grounding.parquet and val_grounding.parquet in the rl folder of the
work directory.
"""
import json, random, re, os, sys
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image

random.seed(2027)
SRC_TRAIN = "./work/agroground_recogneg3_train.jsonl"
SRC_VAL   = "./work/agroground_recogneg3_val.jsonl"
GOLD      = "./work/human_verified_ground_truth_v4.jsonl"
OUT       = "./work/rl"
Q_TRAIN = {"cand_pos": 4000, "cand_pos_dense": 4000, "given_pos": 4000, "given_pos_dense": 4000, "cand_neg": 1500, "given_neg": 1500}
Q_VAL   = {"cand_pos": 100, "cand_pos_dense": 100, "given_pos": 75, "given_pos_dense": 75, "cand_neg": 25, "given_neg": 25}
CLAUSE_FINE   = "Locate all regions affected by {d} in this image. Output bounding boxes in JSON format. If no such regions are present, output an empty list."
CLAUSE_COARSE = "Locate the leaf or plant affected by {d} in this image. Output the bounding box in JSON format. If not present, output an empty list."

def fix_path(p): return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def strip_img(t): return re.sub(r"^<image>\s*", "", t.strip())
def parse_answer(v):
    try:
        data = json.loads(v); boxes, labels = [], []
        for it in data:
            b = it.get("bbox_2d")
            if isinstance(b, list) and len(b) == 4: boxes.append([float(x) for x in b]); labels.append(str(it.get("label", "")))
        lab = Counter(l for l in labels if l).most_common(1)
        return boxes, (lab[0][0] if lab else None)
    except Exception: return None, None
CAND_RE = re.compile(r"one of the following: (.+?)\. Identify")
def parse_cands(t):
    m = CAND_RE.search(t); return [c.strip() for c in m.group(1).split(", ")] if m else None

def bucket_of(ex):
    tf = ex.get("task_format", "")
    if tf in ("recognition", "grounding_given"):
        b, _ = parse_answer(ex["conversations"][1]["value"]); dense = b is not None and len(b) >= 4
        base = "cand_pos" if tf == "recognition" else "given_pos"
        return base + ("_dense" if dense else "")
    if tf == "recognition_negative": return "cand_neg"
    if tf == "grounding_given_negative" and ex.get("phrasing") == "clause": return "given_neg"
    return None

GOLD_PATHS = set(fix_path(json.loads(l)["orig_path"]) for l in open(GOLD))
gold_paths = GOLD_PATHS

def build(src, quotas, split):
    print(f"[{split}] scanning {src} ...", flush=True)
    pools = defaultdict(list)
    with open(src) as f:
        for i, line in enumerate(f):
            ex = json.loads(line); b = bucket_of(ex)
            if b: pools[b].append(ex)
    for b in list(pools): pools[b] = [ex for ex in pools[b] if fix_path(ex["image"]) not in GOLD_PATHS]
    print(f"[{split}] pool sizes (gold images excluded): " + ", ".join(f"{k}={len(v):,}" for k, v in pools.items()), flush=True)
    rows, skipped, idx = [], Counter(), 0
    for b, q in quotas.items():
        pool = pools[b]; random.shuffle(pool); taken = 0
        for ex in pool:
            if taken >= q: break
            instr = strip_img(ex["conversations"][0]["value"]); ans = ex["conversations"][1]["value"]
            healthy = b.endswith("_neg"); fmt = "candidate" if b.startswith("cand") else "given"
            boxes, label = parse_answer(ans)
            if boxes is None: skipped["bad_answer"] += 1; continue
            cands = None
            if fmt == "candidate":
                cands = parse_cands(instr)
                if not cands or (not healthy and (label is None or label not in cands)): skipped["cand_parse"] += 1; continue
            else:
                if not healthy:
                    if not label or not boxes: skipped["no_label"] += 1; continue
                    tpl = CLAUSE_COARSE if ex.get("granularity_level") == "coarse_object" else CLAUSE_FINE
                    instr = tpl.format(d=label)
            if healthy: boxes, label = [], None
            elif not boxes: skipped["no_boxes"] += 1; continue
            path = fix_path(ex["image"])
            try:
                with Image.open(path) as im: W, H = im.size
            except Exception: skipped["missing_image"] += 1; continue
            gt = {"boxes": boxes, "label": label, "candidates": cands, "fmt": fmt, "healthy": healthy}
            rows.append({"data_source": "agroground", "prompt": [{"role": "user", "content": "<image>\n" + instr}],
                         "images": [path], "ability": "grounding",
                         "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                         "extra_info": {"split": split, "index": idx, "id": str(ex.get("source_sample_id", idx)),
                                        "height": H, "width": W, "question": instr, "bucket": b}})
            idx += 1; taken += 1
        print(f"[{split}] {b}: took {taken}/{q}", flush=True)
    random.shuffle(rows)
    for i, r in enumerate(rows): r["extra_info"]["index"] = i
    print(f"[{split}] skipped: {dict(skipped)}", flush=True)
    return rows

train = build(SRC_TRAIN, Q_TRAIN, "train"); val = build(SRC_VAL, Q_VAL, "val")
for name, rows in (("train", train), ("val", val)):
    overlap = sum(1 for r in rows if r["images"][0] in gold_paths)
    assert overlap == 0, f"GOLD OVERLAP in {name}: {overlap}"
    print(f"[{name}] gold-set overlap: 0 (asserted)", flush=True)
smoke = random.sample(train, 32)
# Image-size selection applied to the sampled prompts (not a resize): rows whose image exceeds
# 0.5 megapixels are dropped. At run time this was applied to the written parquet files in
# successive passes (6 MP, 1.5 MP, 0.5 MP); the final 0.5 MP pass is what the trained models saw.
MAX_PIXELS = 500_000
def _within_budget(r): return r["extra_info"]["width"] * r["extra_info"]["height"] <= MAX_PIXELS
_n = {k: len(v) for k, v in (("train", train), ("val", val), ("smoke", smoke))}
train = [r for r in train if _within_budget(r)]; val = [r for r in val if _within_budget(r)]; smoke = [r for r in smoke if _within_budget(r)]
for k, v in (("train", train), ("val", val), ("smoke", smoke)): print(f"[{k}] kept {len(v):,}/{_n[k]:,} rows at <= {MAX_PIXELS/1e6:.1f} MP", flush=True)
os.makedirs(OUT, exist_ok=True)
for name, rows in (("train", train), ("val", val), ("smoke", smoke)):
    pd.DataFrame(rows).to_parquet(f"{OUT}/{name}_grounding.parquet", index=False)
    print(f"wrote {OUT}/{name}_grounding.parquet: {len(rows):,} rows", flush=True)
px = sorted(r["extra_info"]["width"] * r["extra_info"]["height"] for r in train); n = len(px)
print(f"image megapixels: p50 {px[n//2]/1e6:.2f} p90 {px[int(n*.9)]/1e6:.2f} p99 {px[int(n*.99)]/1e6:.2f} max {px[-1]/1e6:.2f}  (~visual tokens max {px[-1]//1024})")
print("\nSAMPLES:")
shown = set()
for r in train:
    b = r["extra_info"]["bucket"]
    if b in shown: continue
    shown.add(b); print(f"\n[{b}] PROMPT: {r['prompt'][0]['content'][:170]!r}\n        GT: {r['reward_model']['ground_truth'][:160]}")
    if len(shown) == 4: break
print("\nRL DATA BUILD DONE")
