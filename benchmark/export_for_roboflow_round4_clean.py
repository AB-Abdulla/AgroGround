"""
Exports an annotation-round manifest for the annotation tool (round 4).

Takes the round's candidate manifest and produces a folder ready for upload to
the annotation tool: all images copied into one folder, plus a COCO-format
_annotations.coco.json carrying the detector's boxes as editable pre-annotations,
so the annotator corrects, deletes and adds boxes rather than drawing from
scratch. The target name is stored as the box label and as the image filename
prefix, so the annotator knows what to look for. Healthy images are included
with no boxes; they are confirmed rather than annotated.

COCO boxes are [x, y, width, height]; pixel boxes are converted and clamped to
the image bounds.

Input: the round's manifest in the work directory. Output: a roboflow_export
folder with the images and _annotations.coco.json.
"""
import json, os, shutil
from PIL import Image

MANIFEST = "./work/eval_round4_manifest.jsonl"
OUTDIR   = "./work/roboflow_export_round4_clean"

def fix_path(p):
    if p and '/home/jovyan' in p:
        p = p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv',
                      './datasets/cddm/images')
    return p

os.makedirs(OUTDIR, exist_ok=True)

# Load manifest
items = []
with open(MANIFEST) as f:
    for line in f:
        items.append(json.loads(line))

# Build COCO structure
coco = {"images": [], "annotations": [], "categories": []}
# One COCO category per disease label, and the disease name also goes into the file name so the
# label is visible in Roboflow.
cat_map = {}
def get_cat_id(name):
    name = (name or "target").strip().lower()
    if name not in cat_map:
        cat_map[name] = len(cat_map)
        coco["categories"].append({"id": cat_map[name], "name": name, "supercategory": "disease"})
    return cat_map[name]

ann_id = 1
copied = 0
skipped = 0

for img_id, m in enumerate(items, start=4000):
    src = fix_path(m["image_path"])
    if not os.path.exists(src):
        skipped += 1
        continue
    # Get dimensions
    W = m.get("image_width"); H = m.get("image_height")
    if not W or not H:
        try:
            with Image.open(src) as im: W, H = im.size
        except:
            skipped += 1
            continue

    # Build a clear filename: regime_dataset_diseaselabel_index.ext
    ext = os.path.splitext(src)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        ext = '.jpg'
    label = (m.get("raw_label") or "healthy").strip().lower().replace(" ", "-").replace("/", "-")[:30]
    fname = f'{m["regime"]}__{m["dataset"]}__{label}__{img_id:04d}{ext}'
    dst = os.path.join(OUTDIR, fname)
    try:
        shutil.copy(src, dst)
    except:
        skipped += 1
        continue
    copied += 1

    coco["images"].append({"id": img_id, "file_name": fname, "width": W, "height": H})

    # Add GDino boxes as pre-annotations (skip for target_absent — empty GT)
    boxes = m.get("gdino_boxes") or []
    disease = m.get("raw_label") or "target"
    for b in boxes:
        x1, y1, x2, y2 = b
        # clamp to bounds
        x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        if w < 1 or h < 1:
            continue
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id,
            "category_id": get_cat_id(disease),
            "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
        })
        ann_id += 1

# Write COCO json
with open(os.path.join(OUTDIR, "_annotations.coco.json"), "w") as f:
    json.dump(coco, f)

print(f"{'='*60}\nROBOFLOW EXPORT COMPLETE\n{'='*60}")
print(f"Images copied : {copied}")
print(f"Skipped       : {skipped}")
print(f"Annotations   : {len(coco['annotations'])} (GDino pre-annotations)")
print(f"Categories    : {len(coco['categories'])} disease types")
print(f"\nOutput folder: {OUTDIR}/")
print(f"  -> contains {copied} images + _annotations.coco.json")
print("\nExport complete: upload the folder to the annotation tool.")
