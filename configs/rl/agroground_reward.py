"""
Composite grounded reward for the GRPO stage (verl custom_reward_function interface).

compute_score(solution_str, ground_truth, extra_info, **kwargs) -> float

The reward is a format gate times a weighted sum of four terms, all scored against
the weak labels of the training example:
  format          0.10   the output parses as a JSON box list
  localization    0.35   mean of matched instance F1 and region IoU
  multi-instance  0.15   recall of the labeled instances
  recognition     0.25   on candidate prompts: 1 for the correct name, 0.25 for a listed
                         but wrong one, 0 otherwise; on known-target prompts this weight
                         is redistributed to localization and multi-instance
  abstention      0.15   two-sided: empty output on healthy examples, non-empty on positives
Healthy examples are scored by the gate and the abstention term only. The weights can
be overridden through reward_kwargs (used by the recognition-weighted ablation).

ground_truth is a JSON string: {"boxes": [[x1, y1, x2, y2], ...] on the 0-1000 grid,
"label": str or null, "candidates": list or null, "fmt": "given" or "candidate",
"healthy": bool}.
"""
import json, math, re
from collections import Counter

W_FORMAT, W_LOC, W_MULTI, W_RECOG, W_ABST = 0.10, 0.35, 0.15, 0.25, 0.15
IOU_THR = 0.5
DEGENERATE_AREA_FRAC = 0.001
RASTER = 100

def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def parse_output(solution_str):
    if not isinstance(solution_str, str): return [], [], False
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", solution_str.strip()).strip()
    try: data = json.loads(s)
    except Exception: return [], [], False
    if not isinstance(data, list): return [], [], False
    boxes, labels = [], []
    for item in data:
        if not isinstance(item, dict) or "bbox_2d" not in item: return [], [], False
        b = item["bbox_2d"]
        if not (isinstance(b, (list, tuple)) and len(b) == 4): return [], [], False
        try: x1, y1, x2, y2 = [float(v) for v in b]
        except Exception: return [], [], False
        if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)): return [], [], False
        x1, x2 = sorted((min(max(x1, 0), 1000), min(max(x2, 0), 1000)))
        y1, y2 = sorted((min(max(y1, 0), 1000), min(max(y2, 0), 1000)))
        boxes.append([x1, y1, x2, y2]); labels.append(_norm(item.get("label", "")))
    return boxes, labels, True

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def greedy_match(preds, gts, thr=IOU_THR):
    if not preds or not gts: return 0
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    up, ug, tp = set(), set(), 0
    for s, i, j in pairs:
        if s < thr: break
        if i in up or j in ug: continue
        up.add(i); ug.add(j); tp += 1
    return tp

def region_iou(preds, gts, grid=RASTER):
    def mask(boxes):
        m = [[False] * grid for _ in range(grid)]
        for x1, y1, x2, y2 in boxes:
            c1, c2 = int(x1 / 1000 * grid), int(math.ceil(x2 / 1000 * grid))
            r1, r2 = int(y1 / 1000 * grid), int(math.ceil(y2 / 1000 * grid))
            for r in range(max(0, r1), min(grid, r2)):
                row = m[r]
                for c in range(max(0, c1), min(grid, c2)): row[c] = True
        return m
    pm, gm = mask(preds), mask(gts); inter = un = 0
    for r in range(grid):
        pr, gr = pm[r], gm[r]
        for c in range(grid):
            if pr[c] and gr[c]: inter += 1
            if pr[c] or gr[c]: un += 1
    return inter / un if un > 0 else 0.0

def _drop_degenerate(boxes, labels):
    kb, kl, nd = [], [], 0
    for b, l in zip(boxes, labels):
        if (b[2] - b[0]) * (b[3] - b[1]) / 1e6 < DEGENERATE_AREA_FRAC: nd += 1
        else: kb.append(b); kl.append(l)
    return kb, kl, nd

def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    global W_FORMAT, W_LOC, W_MULTI, W_RECOG, W_ABST
    W_FORMAT = float(kwargs.get("w_format", W_FORMAT)); W_LOC = float(kwargs.get("w_loc", W_LOC)); W_MULTI = float(kwargs.get("w_multi", W_MULTI))
    W_RECOG = float(kwargs.get("w_recog", W_RECOG)); W_ABST = float(kwargs.get("w_abst", W_ABST))
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    healthy = bool(gt.get("healthy", False)); fmt = gt.get("fmt", "given")
    gt_boxes = [[float(v) for v in b] for b in (gt.get("boxes") or [])]
    true_label = _norm(gt.get("label")); candidates = [_norm(c) for c in (gt.get("candidates") or [])]
    boxes, labels, ok = parse_output(solution_str)
    if not ok: return 0.0
    empty = len(boxes) == 0
    abst = (1.0 if empty else 0.0) if healthy else (0.0 if empty else 1.0)
    if healthy: return W_FORMAT + (1.0 - W_FORMAT) * abst
    mboxes, mlabels, n_degen = _drop_degenerate(boxes, labels)
    tp = greedy_match(mboxes, gt_boxes); n_pred = len(mboxes) + n_degen
    prec = tp / n_pred if n_pred else 0.0; rec = tp / len(gt_boxes) if gt_boxes else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rio = region_iou(mboxes, gt_boxes) if (mboxes and gt_boxes) else 0.0
    loc, multi = 0.5 * (f1 + rio), rec
    if fmt == "candidate":
        w_loc, w_multi, w_recog = W_LOC, W_MULTI, W_RECOG
        maj = Counter(l for l in labels if l).most_common(1); maj = maj[0][0] if maj else ""
        recog = 1.0 if (maj and maj == true_label) else (0.25 if (maj and maj in candidates) else 0.0)
    else:
        scale = (W_LOC + W_MULTI + W_RECOG) / (W_LOC + W_MULTI)
        w_loc, w_multi, w_recog, recog = W_LOC * scale, W_MULTI * scale, 0.0, 0.0
    return float(max(0.0, min(1.0, W_FORMAT + w_loc * loc + w_multi * multi + w_recog * recog + W_ABST * abst)))

if __name__ == "__main__":
    g = lambda b, l, c=None, f="given", h=False: json.dumps({"boxes": b, "label": l, "candidates": c, "fmt": f, "healthy": h})
    G = [[100, 100, 300, 300]]; C = ["rust", "blight", "mildew", "canker"]
    checks = [("perfect given", compute_score('[{"bbox_2d":[100,100,300,300],"label":"rust"}]', g(G, "rust")), 1.0),
              ("empty on diseased", compute_score("[]", g(G, "rust")), 0.10),
              ("healthy empty", compute_score("[]", g([], None, C, "candidate", True)), 1.0),
              ("healthy boxed", compute_score('[{"bbox_2d":[1,1,9,9],"label":"rust"}]', g([], None, C, "candidate", True)), 0.10),
              ("garbage", compute_score("rust everywhere", g(G, "rust")), 0.0)]
    ok = all(abs(a - b) < 1e-6 for _, a, b in checks)
    for n, a, b in checks: print(f"  {n}: {a:.3f} (expect {b:.3f})")
    print("REWARD SELF-TEST:", "PASS" if ok else "FAIL")
