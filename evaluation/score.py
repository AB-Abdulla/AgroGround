"""
Scores the AgroGround prediction files and reproduces the paper's tables.

Everything is computed from two kinds of inputs: the benchmark annotations
(human_verified_ground_truth_v4.jsonl) and the prediction files written by the
inference scripts (one JSON list per model and protocol, keyed by coco_id). Nothing
is recomputed from images.

Metrics (see the paper's metric appendix):
  * localization: greedy one-to-one matching at IoU 0.5, in descending IoU order,
    giving set precision, recall and F1, micro-pooled over the positive images;
    region IoU is the IoU between the unions of predicted and ground-truth boxes
    (rasterized at quarter resolution); lesion-level and object-level F1 are the
    same pooled F1 restricted to those images; an empty prediction scores region
    IoU 0 and counts as false abstention on a positive image
  * recognition (candidate protocol): the predicted label (the most frequent label
    in the output, or the single candidate named in the raw output) equals the
    correct candidate; "none", an off-list name or an unparseable output on a
    positive image is incorrect; joint accuracy is correct recognition with region
    IoU >= 0.5; the first-pick share is the fraction of in-list answers that chose
    the first listed candidate
  * abstention: empty output on the 505 healthy images; false abstention: empty
    output on the 975 positive images
  * intervals: 95% percentile bootstrap over images, 1,000 resamples, a fresh
    numpy generator seeded with 0 for every statistic; paired differences align
    the two models' images by coco_id (ascending), resample the same images for
    both, and report the observed difference with the bootstrap interval (the
    bootstrap mean is printed as well)

Usage:
    python score.py --pred_dir <folder with the prediction files and the gold file>
                    [--plantseg_gt plantseg_test_gt.jsonl] [--sections main,recog,paired,...]

Sections: main (Table 2, known-target protocol), recog (candidate protocol),
paired (paired differences), existence, seeds, scaling, rl (RL ablations),
variants (metric variants), persource (per-source appendix), openset (open-set
naming appendix), plantseg (external transfer), controls. Default: all. Sections
whose prediction files are absent print "(missing: ...)" and continue.

The prediction file names below are the ones written by run_batch.sh,
run_controls.sh and the per-model commands listed in the README.
"""
import os
import re
import json
import argparse
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- file map
GIVEN = [  # (label, file, box key)
    ("GroundingDINO, raw prompts", "gdino_given_v4.json", "pred_boxes_pixel"),
    ("GroundingDINO, taxonomy prompts", "gdino_given_tax_v4.json", "pred_boxes_pixel"),
    ("GroundingDINO, annotator configuration", "gdino_given_tax_v4.json", "pred_boxes_pixel+filters"),
    ("VisionReasoner-7B", "vr_given_v3.json", "pred_boxes_pixel"),
    ("Seg-Zero-7B", "segzero_given_v3.json", "pred_boxes_pixel"),
    ("Qwen3-VL-2B zero-shot", "qwen_ZERO_given_v3.json", "pred_boxes_norm"),
    ("Qwen3-VL-8B zero-shot", "qwen_ZERO8B_given_v3.json", "pred_boxes_norm"),
    ("Ours 2B, grounding only", "qwen_A_given_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed", "qwen_B2_given_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy (6%)", "qwen_C3v2_given_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy (12%)", "qwen_C2v2_given_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy + RL (step 170)", "qwen_RL170_given_v3.json", "pred_boxes_norm"),
    ("RL final checkpoint (step 292)", "qwen_RLfinal_given_v3.json", "pred_boxes_norm"),
    ("RL recognition-weighted reward", "qwen_RLB_given_v3.json", "pred_boxes_norm"),
    ("RL uniform rollout sampling", "qwen_RLC_given_v4.json", "pred_boxes_norm"),
    ("Ours 8B, mixed + healthy (12%)", "qwen_SFT8B_given_v4.json", "pred_boxes_norm"),
    ("12% model, seed 1", "qwen_C2s1_given_v4.json", "pred_boxes_norm"),
    ("12% model, seed 2", "qwen_C2s2_given_v4.json", "pred_boxes_norm"),
    ("PlantSeg expert labels (9,120)", "qwen_PS_given_v4.json", "pred_boxes_norm"),
    ("Pseudo-labels, equal size (9,120)", "qwen_PEQ_given_v4.json", "pred_boxes_norm"),
    ("Pseudo-labels, 10K", "qwen_S10K_given_v4.json", "pred_boxes_norm"),
    ("Pseudo-labels, 100K", "qwen_S100K_given_v4.json", "pred_boxes_norm"),
]
RECOG = [
    ("GroundingDINO (adapted)", "gdino_recog_v4.json", "pred_boxes_pixel"),
    ("VisionReasoner-7B (adapted)", "vr_recog_v3.json", "pred_boxes_pixel"),
    ("Qwen3-VL-2B zero-shot", "qwen_ZERO_recog_v3.json", "pred_boxes_norm"),
    ("Qwen3-VL-8B zero-shot", "qwen_ZERO8B_recog_v3.json", "pred_boxes_norm"),
    ("Ours 2B, grounding only", "qwen_A_recog_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed", "qwen_B2_recog_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy (6%)", "qwen_C3v2_recog_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy (12%)", "qwen_C2v2_recog_v3.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy + RL (step 170)", "qwen_RL170_recog_v3.json", "pred_boxes_norm"),
    ("RL final checkpoint (step 292)", "qwen_RLfinal_recog_v3.json", "pred_boxes_norm"),
    ("RL recognition-weighted reward", "qwen_RLB_recog_v3.json", "pred_boxes_norm"),
    ("RL uniform rollout sampling", "qwen_RLC_recog_v4.json", "pred_boxes_norm"),
    ("Ours 8B, mixed + healthy (12%)", "qwen_SFT8B_recog_v4.json", "pred_boxes_norm"),
    ("12% model, seed 1", "qwen_C2s1_recog_v4.json", "pred_boxes_norm"),
    ("12% model, seed 2", "qwen_C2s2_recog_v4.json", "pred_boxes_norm"),
]
PLANTSEG = [
    ("GroundingDINO, raw prompts", "gdino_given_plantseg.json", "pred_boxes_pixel"),
    ("GroundingDINO, taxonomy prompts", "gdino_given_tax_plantseg.json", "pred_boxes_pixel"),
    ("GroundingDINO, annotator configuration", "gdino_given_tax_plantseg.json", "pred_boxes_pixel+filters"),
    ("Qwen3-VL-2B zero-shot", "qwen_ZERO_given_ps.json", "pred_boxes_norm"),
    ("Qwen3-VL-8B zero-shot", "qwen_ZERO8B_given_ps.json", "pred_boxes_norm"),
    ("Ours 2B, grounding only", "qwen_A_given_ps.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy (12%)", "qwen_C2v2_given_ps.json", "pred_boxes_norm"),
    ("Ours 2B, mixed + healthy + RL (step 170)", "qwen_RL170_given_ps.json", "pred_boxes_norm"),
    ("Ours 8B, mixed + healthy (12%)", "qwen_SFT8B_given_ps.json", "pred_boxes_norm"),
    ("PlantSeg expert labels (9,120)", "qwen_PS_given_ps.json", "pred_boxes_norm"),
    ("Pseudo-labels, equal size (9,120)", "qwen_PEQ_given_ps.json", "pred_boxes_norm"),
]
CONTROLS = {
    "blank": [("Qwen3-VL-2B zero-shot", "ctl_blank_ZERO.json"), ("Ours 2B, mixed", "ctl_blank_B2.json"),
              ("Ours 2B, mixed + healthy (12%)", "ctl_blank_C2.json"), ("RL (step 170)", "ctl_blank_RL.json")],
    "shuffle": [("Ours 2B, mixed", "ctl_shuf_B2.json"), ("Ours 2B, mixed + healthy (12%)", "ctl_shuf_C2.json"),
                ("RL (step 170)", "ctl_shuf_RL.json")],
    "hardneg": [("Ours 2B, mixed", "ctl_hardneg_B2.json"), ("Ours 2B, mixed + healthy (12%)", "ctl_hardneg_C2.json"),
                ("RL (step 170)", "ctl_hardneg_RL.json")],
}

# ---------------------------------------------------------------- helpers (as in the scoring cells)
def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def to_px(r, key):
    bs = r.get(key) or []
    if key == "pred_boxes_norm":
        W, H = r["image_width"], r["image_height"]
        return [[b[0] / 1000 * W, b[1] / 1000 * H, b[2] / 1000 * W, b[3] / 1000 * H] for b in bs]
    return [list(map(float, b)) for b in bs]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def match(preds, gts, thr=0.5):
    """Greedy one-to-one matching in descending IoU order. Returns (tp, fp, fn)."""
    if not gts:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gts)
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)), reverse=True)
    up, ug, tp = set(), set(), 0
    for s, pi, gi in pairs:
        if s < thr:
            break
        if pi in up or gi in ug:
            continue
        up.add(pi)
        ug.add(gi)
        tp += 1
    return tp, len(preds) - len(up), len(gts) - len(ug)


def region_iou(preds, gts, W, H, scale=0.25):
    def mask(boxes):
        w, h = max(1, int(W * scale)), max(1, int(H * scale))
        m = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(int(x1 * scale), w)); x2 = max(0, min(int(x2 * scale), w))
            y1 = max(0, min(int(y1 * scale), h)); y2 = max(0, min(int(y2 * scale), h))
            if x2 > x1 and y2 > y1:
                m[y1:y2, x1:x2] = True
        return m
    pm, gm = mask(preds), mask(gts)
    un = np.logical_or(pm, gm).sum()
    return (np.logical_and(pm, gm).sum() / un) if un > 0 else 0.0


def annotator_filters(boxes, W, H):
    """The annotation pipeline's geometry filters with granularity derived from the boxes
    themselves: area ratio within [0.001, 0.95]; if the largest surviving box covers more
    than half the image the sample is treated as object-level and that single box is kept;
    otherwise NMS at IoU 0.5 over the surviving boxes."""
    A = W * H
    keep = [b for b in boxes if 0.001 <= (b[2] - b[0]) * (b[3] - b[1]) / A <= 0.95]
    if not keep:
        return []
    keep = sorted(keep, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    if (keep[0][2] - keep[0][0]) * (keep[0][3] - keep[0][1]) / A > 0.5:
        return keep[:1]
    out = []
    for b in keep:
        if all(iou(b, o) < 0.5 for o in out):
            out.append(b)
    return out


def boxes_of(r, key):
    if key == "pred_boxes_pixel+filters":
        return annotator_filters(to_px(r, "pred_boxes_pixel"), r["image_width"], r["image_height"])
    return to_px(r, key)


def pred_label(r, key):
    if key != "pred_boxes_norm":
        return norm(r.get("pred_label")) if r.get("pred_label") else None
    labels = [l for l in r.get("pred_labels", []) if l]
    if labels:
        return norm(Counter(labels).most_common(1)[0][0])
    raw = norm(r.get("raw_output", ""))
    hits = [c for c in r.get("candidates", []) if norm(c) in raw]
    return norm(hits[0]) if len(hits) == 1 else None


def f1_of(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


# per-image records and aggregation
def per_image_given(rows, key):
    P = []  # (is_healthy, empty, tp, fp, fn, riou, gran, nbox, gt_n, per-image f1, tp@0.75, fp@0.75, fn@0.75)
    for r in rows:
        px = boxes_of(r, key)
        h = r["granularity"] == "healthy"
        if h:
            P.append((1, int(len(px) == 0), 0, 0, 0, 0.0, "healthy", len(px), 0, 0.0, 0, 0, 0, r.get("dataset"), r["coco_id"]))
            continue
        tp, fp, fn = match(px, r["gt_boxes"])
        t75, f75, n75 = match(px, r["gt_boxes"], 0.75)
        ri = region_iou(px, r["gt_boxes"], r["image_width"], r["image_height"])
        P.append((0, int(len(px) == 0), tp, fp, fn, ri, r["granularity"], len(px), len(r["gt_boxes"]),
                  f1_of(tp, fp, fn)[2], t75, f75, n75, r.get("dataset"), r["coco_id"]))
    return P


def agg(P, idx=None):
    S = [P[i] for i in idx] if idx is not None else P
    pos = [s for s in S if not s[0]]
    hea = [s for s in S if s[0]]
    tp = sum(s[2] for s in pos); fp = sum(s[3] for s in pos); fn = sum(s[4] for s in pos)
    p, r, f = f1_of(tp, fp, fn)
    def sub(pred):
        q = [sum(s[i] for s in pos if pred(s)) for i in (2, 3, 4)]
        return f1_of(*q)[2]
    dense = [s for s in pos if s[6] == "fine" and s[8] >= 5]
    return {"P": p, "R": r, "F1": f,
            "RIoU": np.mean([s[5] for s in pos]) if pos else 0.0,
            "abst": np.mean([s[1] for s in hea]) if hea else 0.0,
            "FA": np.mean([s[1] for s in pos]) if pos else 0.0,
            "fineF1": sub(lambda s: s[6] == "fine"), "coarseF1": sub(lambda s: s[6] == "coarse"),
            "denseF1": sub(lambda s: s[6] == "fine" and s[8] >= 5), "nondenseF1": sub(lambda s: not (s[6] == "fine" and s[8] >= 5)),
            "macroF1": np.mean([s[9] for s in pos]) if pos else 0.0,
            "F1@0.75": f1_of(sum(s[10] for s in pos), sum(s[11] for s in pos), sum(s[12] for s in pos))[2],
            "boxes": np.mean([s[7] for s in pos if s[6] == "fine"]) if any(s[6] == "fine" for s in pos) else 0.0,
            "boxes_dense": np.mean([s[7] for s in dense]) if dense else 0.0}


def per_image_recog(rows, key):
    P = []  # (is_healthy, empty, correct, joint, pos_idx, offlist, instance_joint, fine)
    for r in rows:
        ml = pred_label(r, key)
        px = boxes_of(r, key)
        h = r["granularity"] == "healthy"
        if h:
            P.append((1, int(len(px) == 0), 0, 0, -1, 0, 0, 0, r.get("dataset"), r["coco_id"]))
            continue
        cn = [norm(c) for c in r["candidates"]]
        cr = norm(r["correct_answer"])
        cor = int(ml == cr)
        pos = cn.index(ml) if ml in cn else -1
        off = int(bool(ml) and ml not in cn)
        ri = region_iou(px, r["gt_boxes"], r["image_width"], r["image_height"])
        tp, _, _ = match(px, r["gt_boxes"])
        P.append((0, int(len(px) == 0), cor, int(cor and ri >= 0.5), pos, off, int(cor and tp >= 1),
                  int(r["granularity"] == "fine"), r.get("dataset"), r["coco_id"]))
    return P


def agg_recog(P, idx=None):
    S = [P[i] for i in idx] if idx is not None else P
    pos = [s for s in S if not s[0]]
    hea = [s for s in S if s[0]]
    inlist = [s for s in pos if s[4] >= 0]
    fine = [s for s in pos if s[7]]
    return {"acc": np.mean([s[2] for s in pos]) if pos else 0.0,
            "joint": np.mean([s[3] for s in pos]) if pos else 0.0,
            "pos1": np.mean([s[4] == 0 for s in inlist]) if inlist else 0.0,
            "offlist": sum(s[5] for s in pos),
            "abst": np.mean([s[1] for s in hea]) if hea else 0.0,
            "FA": np.mean([s[1] for s in pos]) if pos else 0.0,
            "instance_joint": np.mean([s[6] for s in pos]) if pos else 0.0,
            "fine_joint": np.mean([s[3] for s in fine]) if fine else 0.0}


# bootstrap helpers: a fresh generator seeded with 0 for every statistic
def boot(fn, n, B=1000, seed=0):
    g = np.random.default_rng(seed)
    v = [fn(g.integers(0, n, n)) for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def paired(PA, PB, aggf, key, B=1000, seed=0):
    """Paired bootstrap over the same images, aligned by coco_id (ascending);
    returns (observed, bootstrap mean, lo, hi)."""
    A = {s[-1]: s for s in PA}
    Bd = {s[-1]: s for s in PB}
    ids = sorted(set(A) & set(Bd))
    PA = [A[i] for i in ids]
    PB = [Bd[i] for i in ids]
    g = np.random.default_rng(seed)
    n = len(PA)
    d = []
    for _ in range(B):
        idx = g.integers(0, n, n)
        d.append(aggf(PB, idx)[key] - aggf(PA, idx)[key])
    return aggf(PB)[key] - aggf(PA)[key], float(np.mean(d)), np.percentile(d, 2.5), np.percentile(d, 97.5)


def positives(P):
    return [s for s in P if not s[0]]


def healthy(P):
    return [s for s in P if s[0]]


# ---------------------------------------------------------------- loading
def load_rows(path, ids, gt=None):
    """Rows of a prediction file restricted to the evaluation ids, in file order. If a row
    lacks gt_boxes (e.g. PlantSeg predictions released without the derived ground truth),
    the ground truth is joined from the gold file by coco_id."""
    rows = [r for r in json.load(open(path)) if r["coco_id"] in ids]
    if gt is not None:
        for r in rows:
            if "gt_boxes" not in r:
                r["gt_boxes"] = gt[r["coco_id"]]["gt_boxes"]
            for k in ("granularity", "image_width", "image_height"):
                if k not in r:
                    r[k] = gt[r["coco_id"]][k]
    return rows


def fmt_ci(v, lo, hi, pct=False):
    if pct:
        return f"{v * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- sections
def run(args):
    D = args.pred_dir
    gold = {}
    for l in open(os.path.join(D, args.gold)):
        r = json.loads(l)
        gold[r["coco_id"]] = r
    ids = set(gold)
    if args.exclude:
        drop = {int(x) for x in args.exclude.split(",") if x.strip()}
        ids -= drop
        gold = {k: v for k, v in gold.items() if k in ids}
        print(f"excluding {len(drop)} image(s): {sorted(drop)}")
    want = set(args.sections.split(",")) if args.sections != "all" else None
    def on(s):
        return want is None or s in want

    PG, PR = {}, {}
    def given(name):
        if name in PG:
            return PG[name]
        for lab, f, key in GIVEN:
            if lab == name:
                p = os.path.join(D, f)
                if not os.path.exists(p):
                    print(f"  (missing: {f})"); PG[name] = None; return None
                PG[name] = per_image_given(load_rows(p, ids, gold), key)
                return PG[name]
        raise KeyError(name)
    def recog(name):
        if name in PR:
            return PR[name]
        for lab, f, key in RECOG:
            if lab == name:
                p = os.path.join(D, f)
                if not os.path.exists(p):
                    print(f"  (missing: {f})"); PR[name] = None; return None
                PR[name] = per_image_recog(load_rows(p, ids, gold), key)
                return PR[name]
        raise KeyError(name)
    def both(*names):
        """True when every named prediction set is available."""
        return all(x is not None for x in names)

    n_pos = sum(1 for r in gold.values() if r["granularity"] != "healthy")
    n_h = len(gold) - n_pos
    print(f"gold: {len(gold)} images ({n_pos} positive, {n_h} healthy)")

    if on("main"):
        print("\n=== Table 2, known-target protocol (F1 and abstention with 95% CI; seed 0, B=1000) ===")
        print(f"{'model':<42} {'F1 [CI]':<22} {'R':>6} {'P':>6} {'RIoU':>6} {'fine':>6} {'coarse':>7} {'abst [CI]':<22} {'FA':>6} {'boxes':>6}")
        for lab, f, key in GIVEN[:16]:
            if not os.path.exists(os.path.join(D, f)):
                print(f"{lab:<42} (missing: {f})"); continue
            P = given(lab); g = agg(P); n = len(P)
            f1 = boot(lambda i: agg(P, i)["F1"], n); ab = boot(lambda i: agg(P, i)["abst"], n)
            print(f"{lab:<42} {fmt_ci(g['F1'], *f1):<22} {g['R']:6.3f} {g['P']:6.3f} {g['RIoU']:6.3f} {g['fineF1']:6.3f} {g['coarseF1']:7.3f} "
                  f"{fmt_ci(g['abst'], *ab, pct=True):<22} {g['FA'] * 100:5.1f}% {g['boxes']:6.2f}")

    if on("recog"):
        print("\n=== Candidate protocol (recognition and joint accuracy with 95% CI; seed 0, B=1000) ===")
        print(f"{'model':<42} {'recognition [CI]':<22} {'first-pick':>10} {'joint [CI]':<22} {'abst':>6} {'FA':>6} {'off-list':>8}")
        for lab, f, key in RECOG:
            if not os.path.exists(os.path.join(D, f)):
                print(f"{lab:<42} (missing: {f})"); continue
            P = recog(lab); r = agg_recog(P); n = len(P)
            a = boot(lambda i: agg_recog(P, i)["acc"], n); j = boot(lambda i: agg_recog(P, i)["joint"], n)
            print(f"{lab:<42} {fmt_ci(r['acc'], *a, pct=True):<22} {r['pos1'] * 100:9.1f}% {fmt_ci(r['joint'], *j, pct=True):<22} "
                  f"{r['abst'] * 100:5.1f}% {r['FA'] * 100:5.1f}% {r['offlist']:8d}")

    if on("paired"):
        print("\n=== Paired differences (same images; observed difference, bootstrap mean, 95% CI; seed 0, B=1000) ===")
        C2, RL, A, B2 = "Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy + RL (step 170)", "Ours 2B, grounding only", "Ours 2B, mixed"
        pairs = [(C2, RL), (A, B2), ("GroundingDINO, raw prompts", C2), ("GroundingDINO, taxonomy prompts", C2),
                 ("GroundingDINO, raw prompts", RL), ("GroundingDINO, taxonomy prompts", RL),
                 ("VisionReasoner-7B", A), ("VisionReasoner-7B", RL), ("Seg-Zero-7B", A), ("Seg-Zero-7B", C2), ("Seg-Zero-7B", RL),
                 (C2, "Ours 8B, mixed + healthy (12%)"),
                 ("PlantSeg expert labels (9,120)", "Pseudo-labels, equal size (9,120)"), ("Pseudo-labels, equal size (9,120)", A),
                 ("Pseudo-labels, 10K", A)]
        print(f"F1, resampling all benchmark images (n = {len(gold):,}):")
        for a, b in pairs:
            PA, PB = given(a), given(b)
            if not both(PA, PB):
                continue
            obs, mean, lo, hi = paired(PA, PB, agg, "F1")
            print(f"  {b} - {a}: {obs:+.3f} (boot mean {mean:+.3f}) [{lo:+.3f}, {hi:+.3f}]")
        print(f"F1 against the annotator configuration, resampling positive images (n = {n_pos:,}):")
        AN = "GroundingDINO, annotator configuration"
        for b in (C2, RL):
            if both(given(AN), given(b)):
                obs, mean, lo, hi = paired(positives(given(AN)), positives(given(b)), agg, "F1")
                print(f"  {b} - {AN}: {obs:+.3f} (boot mean {mean:+.3f}) [{lo:+.3f}, {hi:+.3f}]")
        print(f"RL (step 170) minus the 12% model, per-image statistics (n = {n_pos:,} positives, or {n_h:,} healthy):")
        if both(recog(C2), recog(RL)):
            pc, pr = positives(recog(C2)), positives(recog(RL))
            for key, label in (("acc", "recognition"), ("joint", "joint accuracy")):
                obs, mean, lo, hi = paired(pc, pr, agg_recog, key)
                print(f"  {label}: {obs * 100:+.1f} pts (boot mean {mean * 100:+.1f}) [{lo * 100:+.1f}, {hi * 100:+.1f}]")
        if both(given(C2), given(RL)):
            hc, hr = healthy(given(C2)), healthy(given(RL))
            obs, mean, lo, hi = paired(hc, hr, agg, "abst")
            print(f"  healthy-image abstention (known-target): {obs * 100:+.1f} pts (boot mean {mean * 100:+.1f}) [{lo * 100:+.1f}, {hi * 100:+.1f}]")
            obs, mean, lo, hi = paired(positives(given(C2)), positives(given(RL)), agg, "R")
            print(f"  recall: {obs:+.3f} (boot mean {mean:+.3f}) [{lo:+.3f}, {hi:+.3f}]")
        if both(given("Pseudo-labels, 10K"), given(A)):
            obs, mean, lo, hi = paired(positives(given("Pseudo-labels, 10K")), positives(given(A)), agg, "F1")
            print(f"  full corpus minus 10K subset, F1 (positives): {obs:+.3f} (boot mean {mean:+.3f}) [{lo:+.3f}, {hi:+.3f}]")
        print("8B minus 12% 2B, candidate protocol (all images, and positives only):")
        B8 = "Ours 8B, mixed + healthy (12%)"
        if both(recog(C2), recog(B8)):
            for key, label in (("acc", "recognition"), ("joint", "joint accuracy")):
                obs, mean, lo, hi = paired(recog(C2), recog(B8), agg_recog, key)
                o2, m2, l2, h2 = paired(positives(recog(C2)), positives(recog(B8)), agg_recog, key)
                print(f"  {label}: {obs * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}] (all images); [{l2 * 100:+.1f}, {h2 * 100:+.1f}] (positives only)")

    if on("existence"):
        print("\n=== Existence-aware supervision (healthy abstention, false abstention, grounding, recognition) ===")
        for share, name in (("0%", "Ours 2B, mixed"), ("6%", "Ours 2B, mixed + healthy (6%)"), ("12%", "Ours 2B, mixed + healthy (12%)")):
            P = given(name); Rp = recog(name)
            if not both(P, Rp):
                continue
            g = agg(P); ab = boot(lambda i: agg(P, i)["abst"], len(P)); R = agg_recog(Rp)
            print(f"  {share:<4} abst {fmt_ci(g['abst'], *ab, pct=True)}  cand-abst {R['abst'] * 100:.1f}%  FA {g['FA'] * 100:.1f}% / {R['FA'] * 100:.1f}%  "
                  f"R {g['R']:.3f}  F1 {g['F1']:.3f}  RIoU {g['RIoU']:.3f}  recog {R['acc'] * 100:.1f}%  joint {R['joint'] * 100:.1f}%  boxes {g['boxes']:.2f} (dense {g['boxes_dense']:.2f})")

    if on("seeds"):
        print("\n=== Three training seeds of the 12% model (mean +/- sample std) ===")
        names = ["Ours 2B, mixed + healthy (12%)", "12% model, seed 1", "12% model, seed 2"]
        vals = {k: [] for k in ("F1", "R", "abst", "FA", "recog", "joint")}
        for nm in names:
            if not both(given(nm), recog(nm)):
                continue
            g = agg(given(nm)); r = agg_recog(recog(nm))
            vals["F1"].append(g["F1"]); vals["R"].append(g["R"]); vals["abst"].append(g["abst"]); vals["FA"].append(g["FA"])
            vals["recog"].append(r["acc"]); vals["joint"].append(r["joint"])
            print(f"  {nm:<36} F1 {g['F1']:.3f}  R {g['R']:.3f}  abst {g['abst'] * 100:.1f}%  FA {g['FA'] * 100:.1f}%  recog {r['acc'] * 100:.1f}%  joint {r['joint'] * 100:.1f}%")
        for k, v in vals.items():
            if len(v) < 2:
                continue
            v = np.array(v); s = f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if k in ("F1", "R") else f"{v.mean() * 100:.1f} +/- {v.std(ddof=1) * 100:.1f}"
            print(f"  {k}: {s}")

    if on("scaling"):
        print("\n=== Expert labels, pseudo-labels and scale (known-target protocol) ===")
        for nm in ("PlantSeg expert labels (9,120)", "Pseudo-labels, equal size (9,120)", "Pseudo-labels, 10K", "Pseudo-labels, 100K", "Ours 2B, grounding only"):
            P = given(nm)
            if P is None:
                continue
            g = agg(P); f1 = boot(lambda i: agg(P, i)["F1"], len(P))
            print(f"  {nm:<36} F1 {fmt_ci(g['F1'], *f1)}  R {g['R']:.3f}  P {g['P']:.3f}  RIoU {g['RIoU']:.3f}  fine {g['fineF1']:.3f}  coarse {g['coarseF1']:.3f}")

    if on("rl"):
        print("\n=== RL ablations (known-target and candidate protocols) ===")
        for nm in ("Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy + RL (step 170)", "RL final checkpoint (step 292)", "RL recognition-weighted reward", "RL uniform rollout sampling"):
            P = given(nm); Rp = recog(nm)
            if not both(P, Rp):
                continue
            g = agg(P); f1 = boot(lambda i: agg(P, i)["F1"], len(P)); R = agg_recog(Rp)
            print(f"  {nm:<42} F1 {fmt_ci(g['F1'], *f1)}  R {g['R']:.3f}  fine {g['fineF1']:.3f}  boxes(dense) {g['boxes_dense']:.2f}  "
                  f"abst {g['abst'] * 100:.1f}% / {R['abst'] * 100:.1f}%  FA {g['FA'] * 100:.1f}%  recog {R['acc'] * 100:.1f}%  joint {R['joint'] * 100:.1f}%")

    if on("variants"):
        print("\n=== Metric variants (known-target protocol): micro, macro, dense, non-dense, F1@0.75 ===")
        for nm in ("GroundingDINO, raw prompts", "GroundingDINO, taxonomy prompts", "VisionReasoner-7B", "Seg-Zero-7B", "Qwen3-VL-2B zero-shot", "Qwen3-VL-8B zero-shot", "Ours 2B, grounding only", "Ours 2B, mixed", "Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy + RL (step 170)", "Ours 8B, mixed + healthy (12%)"):
            if given(nm) is None:
                continue
            g = agg(given(nm))
            print(f"  {nm:<42} micro {g['F1']:.3f}  macro {g['macroF1']:.3f}  dense {g['denseF1']:.3f}  non-dense {g['nondenseF1']:.3f}  F1@0.75 {g['F1@0.75']:.3f}")
        print("Candidate protocol: joint variants (region IoU >= 0.5, >= 1 matched instance, lesion-level images only)")
        for nm in ("Ours 2B, grounding only", "Ours 2B, mixed", "Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy + RL (step 170)", "Ours 8B, mixed + healthy (12%)"):
            if recog(nm) is None:
                continue
            r = agg_recog(recog(nm))
            print(f"  {nm:<42} joint {r['joint'] * 100:.1f}%  instance {r['instance_joint'] * 100:.1f}%  fine-only {r['fine_joint'] * 100:.1f}%")

    if on("persource"):
        print("\n=== Per-source results (recognition accuracy / known-target F1) ===")
        sources = ("leafnet", "cddm", "mirage", "agrobench", "leafbench", "agromind", "agmmu", "agrocot")
        rows = [("GroundingDINO (raw prompts; adapted recognition)", "GroundingDINO (adapted)", "GroundingDINO, raw prompts"),
                ("GroundingDINO, taxonomy prompts (grounding only)", None, "GroundingDINO, taxonomy prompts"),
                ("Qwen3-VL-8B zero-shot", "Qwen3-VL-8B zero-shot", "Qwen3-VL-8B zero-shot"),
                ("Ours 2B, mixed", "Ours 2B, mixed", "Ours 2B, mixed"),
                ("Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy (12%)", "Ours 2B, mixed + healthy (12%)"),
                ("Ours 2B, mixed + healthy + RL (step 170)", "Ours 2B, mixed + healthy + RL (step 170)", "Ours 2B, mixed + healthy + RL (step 170)"),
                ("Ours 8B, mixed + healthy (12%)", "Ours 8B, mixed + healthy (12%)", "Ours 8B, mixed + healthy (12%)")]
        print(f"{'model':<52}" + "".join(f"{src:>14}" for src in sources))
        for label, rn, gn in rows:
            R = recog(rn) if rn else None; G = given(gn)
            if G is None:
                continue
            cells = []
            for src in sources:
                gsub = [p for p in G if p[-2] == src]; g = agg(gsub) if gsub else None
                rsub = [p for p in R if p[-2] == src] if R else None; r = agg_recog(rsub) if rsub else None
                cells.append(f"{(r['acc'] * 100 if r else float('nan')):5.1f} / {(g['F1'] if g else float('nan')):.3f}" if r else f"   -- / {(g['F1'] if g else float('nan')):.3f}")
            print(f"{label:<52}" + "".join(f"{c:>14}" for c in cells))

    if on("openset"):
        print("\n=== Open-set naming (no candidate list): strict and lenient name match ===")
        files = (("Qwen3-VL-2B zero-shot", "openset2_ZERO_v4.json"), ("Ours 2B, grounding only", "openset2_A_v4.json"),
                 ("Ours 2B, mixed", "openset2_B2_v4.json"), ("Ours 2B, mixed + healthy (12%)", "openset2_C2v2_v4.json"),
                 ("Ours 2B, mixed + healthy + RL (step 170)", "openset2_RL170_v4.json"), ("RL recognition-weighted reward", "openset2_RLB_v4.json"),
                 ("RL uniform rollout sampling", "openset2_RLC_v4.json"),
                 ("[first formulation] Qwen3-VL-2B zero-shot", "openset_ZERO_v4.json"), ("[first formulation] Ours 2B, grounding only", "openset_A_v4.json"),
                 ("[first formulation] Ours 2B, mixed", "openset_B2_v4.json"), ("[first formulation] Ours 2B, mixed + healthy (12%)", "openset_C2v2_v4.json"))
        print(f"{'model':<42} {'format ok':>9} {'strict':>7} {'lenient':>8} {'F1':>6} {'RIoU':>6} {'joint':>6} {'abst':>9} {'FA':>6}")
        for tag, f in files:
            p = os.path.join(D, f)
            if not os.path.exists(p):
                print(f"{tag:<42} (missing: {f})"); continue
            rows = [r for r in json.load(open(p)) if r["coco_id"] in ids]
            n = st = le = joint = 0; I = {"tp": 0, "fp": 0, "fn": 0}; R = []; h = ht = fa = fok = 0
            for r in rows:
                fok += bool(r.get("format_ok")); px = to_px(r, "pred_boxes_norm")
                if r["granularity"] == "healthy":
                    ht += 1; h += (len(px) == 0); continue
                n += 1; fa += (len(px) == 0); g_ = norm(r["gold_label"])
                labels = [l for l in r.get("pred_labels", []) if l]
                ml = norm(Counter(labels).most_common(1)[0][0]) if labels else ""
                strict = (ml == g_ and ml != ""); lenient = strict or (ml != "" and (ml in g_ or g_ in ml))
                st += strict; le += lenient
                tp, fp, fn = match(px, r["gt_boxes"]); I["tp"] += tp; I["fp"] += fp; I["fn"] += fn
                ri = region_iou(px, r["gt_boxes"], r["image_width"], r["image_height"]); R.append(ri); joint += (lenient and ri >= 0.5)
            pr_ = I["tp"] / max(1, I["tp"] + I["fp"]); rc = I["tp"] / max(1, I["tp"] + I["fn"]); f1 = 2 * pr_ * rc / max(1e-9, pr_ + rc)
            print(f"{tag:<42} {fok / max(1, len(rows)) * 100:8.1f}% {st / max(1, n) * 100:6.1f}% {le / max(1, n) * 100:7.1f}% {f1:6.3f} {np.mean(R) if R else 0:6.3f} {joint / max(1, n) * 100:5.1f}% {str(h) + '/' + str(ht):>9} {fa / max(1, n) * 100:5.1f}%")

    if on("plantseg"):
        gt_path = os.path.join(D, args.plantseg_gt) if args.plantseg_gt else None
        psg = None
        if gt_path and os.path.exists(gt_path):
            psg = {json.loads(l)["coco_id"]: json.loads(l) for l in open(gt_path)}
            pids = set(psg)
        else:
            # ids from any PlantSeg prediction file that carries ground truth
            f0 = os.path.join(D, "qwen_RL170_given_ps.json")
            if not os.path.exists(f0):
                print("\n(PlantSeg section skipped: no PlantSeg ground truth or predictions found)"); pids = set()
            else:
                pids = {r["coco_id"] for r in json.load(open(f0))}
        if pids:
            print(f"\n=== PlantSeg test set (n = {len(pids)}), known-target protocol ===")
            PT = {}
            for lab, f, key in PLANTSEG:
                p = os.path.join(D, f)
                if not os.path.exists(p):
                    print(f"  {lab:<42} (missing: {f})"); continue
                P = per_image_given(load_rows(p, pids, psg), key); PT[lab] = P; g = agg(P); f1 = boot(lambda i: agg(P, i)["F1"], len(P))
                print(f"  {lab:<42} F1 {fmt_ci(g['F1'], *f1)}  R {g['R']:.3f}  P {g['P']:.3f}  RIoU {g['RIoU']:.3f}  fine {g['fineF1']:.3f}  coarse {g['coarseF1']:.3f}")
            RLn = "Ours 2B, mixed + healthy + RL (step 170)"
            for a, b in (("Ours 2B, mixed + healthy (12%)", RLn), ("PlantSeg expert labels (9,120)", RLn), ("GroundingDINO, taxonomy prompts", RLn),
                         ("GroundingDINO, annotator configuration", RLn), ("Ours 2B, mixed + healthy (12%)", "Ours 8B, mixed + healthy (12%)"),
                         ("Pseudo-labels, equal size (9,120)", "PlantSeg expert labels (9,120)")):
                if a in PT and b in PT:
                    obs, mean, lo, hi = paired(PT[a], PT[b], agg, "F1")
                    print(f"  {b} - {a}: {obs:+.3f} (boot mean {mean:+.3f}) [{lo:+.3f}, {hi:+.3f}]")

    if on("controls"):
        print("\n=== Controls on the positive images (seed 23 at generation; intervals seed 0) ===")
        def ctl_rows(f):
            return [r for r in json.load(open(os.path.join(D, f))) if r["coco_id"] in ids and r["granularity"] != "healthy"]
        for mode in ("blank", "shuffle"):
            for lab, f in CONTROLS[mode]:
                p = os.path.join(D, f)
                if not os.path.exists(p):
                    print(f"  {mode:<8} {lab:<36} (missing: {f})"); continue
                rows = ctl_rows(f)
                cor = np.array([int(pred_label(r, "pred_boxes_norm") == norm(r["correct_answer"])) for r in rows], float)
                emp = np.mean([len(r.get("pred_boxes_norm") or []) == 0 for r in rows])
                lo, hi = boot(lambda i: cor[i].mean(), len(cor))
                print(f"  {mode:<8} {lab:<36} recognition {cor.mean() * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]  empty output {emp * 100:.1f}%")
        for lab, f in CONTROLS["hardneg"]:
            p = os.path.join(D, f)
            if not os.path.exists(p):
                print(f"  hardneg  {lab:<36} (missing: {f})"); continue
            rows = ctl_rows(f)
            ab = np.array([int(len(r.get("pred_boxes_norm") or []) == 0) for r in rows], float)
            lo, hi = boot(lambda i: ab[i].mean(), len(ab))
            print(f"  hardneg  {lab:<36} abstention on wrong-target queries {ab.mean() * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=".", help="folder with the gold file and the prediction files")
    ap.add_argument("--gold", default="human_verified_ground_truth_v4.jsonl")
    ap.add_argument("--plantseg_gt", default="plantseg_test_gt.jsonl", help="PlantSeg test ground truth (built locally); optional")
    ap.add_argument("--sections", default="all")
    ap.add_argument("--exclude", default="", help="comma-separated coco_ids to leave out; '320,3347' reproduces benchmark v4.1 from the v4 file (the two images withdrawn after submission)")
    run(ap.parse_args())
