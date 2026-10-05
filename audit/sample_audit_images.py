"""
Samples the 100 pseudo-labeled images of the box-quality audit (seed 7).

The sample is drawn from the training records of agroground_granularity.jsonl (the
file written by build_granularity.py): every 200th line is read, records with at
least one box are kept, and they are grouped by (source dataset, granularity level).
The eight largest groups are shuffled with seed 7 and the first 13 of each are taken
(two groups have only 11), giving the 100 images of the audit sample. The resulting
strata are CDDM lesion-level and object-level, LeafNet lesion-level and object-level,
MIRAGE lesion-level and LeafBench object-level (13 each), and AgroBench lesion-level
and LeafBench lesion-level (11 each).

Output: audit_sample.json, one record per image with its index, image_id (the image's
path relative to the source's release root), source, granularity, label and the
pipeline's boxes. The verdicts given to these images
during the audit are in audit_verdicts.json (every box correct; wrong object;
whole-leaf box on a lesion image; missed lesions; duplicate leaf-plus-region boxes).

Running the script with --grids also renders contact sheets (20 images per sheet)
with the boxes drawn, which is what the auditor judged.

Paths: AGROGROUND_WORK_DIR is the work directory (default ./work) and
AGROGROUND_DATA_ROOT the folder holding the source datasets (default ./datasets);
image_id values are paths relative to that folder.
"""
import os
import sys
import json
import math
import random

WORK_DIR = os.environ.get("AGROGROUND_WORK_DIR", "./work")
DATA_ROOT = os.environ.get("AGROGROUND_DATA_ROOT", "./datasets")
GRANULARITY = os.path.join(WORK_DIR, "agroground_granularity.jsonl")
OUTPUT = os.path.join(WORK_DIR, "audit_sample.json")

PATH_REMAP = {"/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv": os.path.join(DATA_ROOT, "cddm", "images")}


def resolve_image_path(p):
    if not p:
        return p
    for src, dst in PATH_REMAP.items():
        if p.startswith(src):
            return dst + p[len(src):]
    return p if os.path.isabs(p) else os.path.join(DATA_ROOT, p)


def image_id(p):
    """Path relative to the source's release root (the released keying), mapping the CDDM prefix."""
    for src in PATH_REMAP:
        if p.startswith(src):
            return "cddm/images" + p[len(src):]
    root = os.path.abspath(DATA_ROOT) + os.sep
    ap = os.path.abspath(p)
    return ap[len(root):] if ap.startswith(root) else p


def sample():
    random.seed(7)
    rows = []
    with open(GRANULARITY) as f:
        for i, l in enumerate(f):
            if i % 200 == 0:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if r.get("boxes"):
                    rows.append(r)
    strata = {}
    for r in rows:
        strata.setdefault((r.get("stageD_dataset"), r.get("granularity_level")), []).append(r)
    out = []
    for k, v in sorted(strata.items(), key=lambda kv: -len(kv[1]))[:8]:
        random.shuffle(v)
        out += v[:13]
    out = out[:100]
    json.dump([{"idx": i, "image_id": image_id(r["image_path"]), "dataset": r.get("stageD_dataset"),
                "granularity": r.get("granularity_level"), "label": r.get("stageD_label"), "boxes": r["boxes"]}
               for i, r in enumerate(out)], open(OUTPUT, "w"))
    counts = {}
    for s in out:
        k = (s.get("stageD_dataset"), s.get("granularity_level"))
        counts[k] = counts.get(k, 0) + 1
    print("audit sample:", len(out), "images; strata:", counts)
    return out


def grids(out, per=20, cols=5):
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    for g in range(math.ceil(len(out) / per)):
        batch = out[g * per:(g + 1) * per]
        nrows = math.ceil(len(batch) / cols)
        fig, axes = plt.subplots(nrows, cols, figsize=(20, 4.2 * nrows))
        for i, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if i < len(batch):
                r = batch[i]
                im = Image.open(resolve_image_path(r["image_path"])).convert("RGB")
                d = ImageDraw.Draw(im)
                W, H = im.size
                for b in r["boxes"]:
                    x1, y1, x2, y2 = b[:4]
                    d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=max(2, int(min(W, H) / 150)))
                ax.imshow(im)
                ax.set_title(f"#{g * per + i}  {r.get('stageD_dataset')} | {str(r.get('granularity_level', ''))[:6]} | {str(r.get('stageD_label'))[:28]}", fontsize=8)
        plt.suptitle(f"Pseudo-label audit grid {g + 1}: judge the green boxes", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(WORK_DIR, f"audit_grid_{g + 1}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    out = sample()
    if "--grids" in sys.argv:
        grids(out)
