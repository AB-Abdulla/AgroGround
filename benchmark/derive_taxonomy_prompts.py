"""
Derives the annotation-time taxonomy prompt for every benchmark image.

Produces gold_taxonomy_prompts_v4.json, the file read by evaluation/gdino_baseline_eval_tax_v4.py
(the "taxonomy prompts" teacher configuration). For each benchmark image, the prompt is
the one the annotation pipeline used for that very image when it exists in
final_gdino_results.jsonl (the image is held out from training, but the pipeline ran on
every groundable sample); otherwise the prompt used for the same normalized label on any
sample; otherwise the raw name. For healthy images the lookup is done on the crop-matched
absent disease. The script prints how many prompts came from each source.

Input (work directory): human_verified_ground_truth_v4.jsonl and final_gdino_results.jsonl.
Output: gold_taxonomy_prompts_v4.json, mapping coco_id to prompt.
final_gdino_results.jsonl is the output of construction step 3 (final_gdino_script.py); it is the
as-run source of the prompts because it records, for every groundable sample, the prompt the
detector actually received.
Paths are relative to the repository root.
"""
import json


def fix(p):
    return p.replace("/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv", "./datasets/cddm/images") if p and "/home/jovyan" in p else p


def main():
    gold = [json.loads(l) for l in open("./work/human_verified_ground_truth_v4.jsonl")]
    paths = {fix(g["orig_path"]) for g in gold}
    prompt_by_path, prompt_by_label = {}, {}
    for line in open("./work/final_gdino_results.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = fix(r.get("image_path", ""))
        pr = r.get("stageE_prompt")
        lab = (r.get("stageD_label") or "").strip().lower()
        if pr and p in paths and p not in prompt_by_path:
            prompt_by_path[p] = pr
        if pr and lab and lab not in prompt_by_label:
            prompt_by_label[lab] = pr
    out = {}
    src = {"own": 0, "label": 0, "raw": 0}
    for g in gold:
        p = fix(g["orig_path"])
        lab = ((g.get("abstention_disease") if g["granularity"] == "healthy" else g.get("disease_raw")) or "").strip().lower()
        if g["granularity"] != "healthy" and p in prompt_by_path:
            out[g["coco_id"]] = prompt_by_path[p]
            src["own"] += 1
        elif lab in prompt_by_label:
            out[g["coco_id"]] = prompt_by_label[lab]
            src["label"] += 1
        else:
            out[g["coco_id"]] = lab
            src["raw"] += 1
    json.dump(out, open("./work/gold_taxonomy_prompts_v4.json", "w"))
    print("taxonomy prompts for the benchmark:", src, "| example:", next(iter(out.values()))[:100])


if __name__ == "__main__":
    main()
