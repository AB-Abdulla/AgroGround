"""
Derives the taxonomy prompt for every PlantSeg test image (nothing from PlantSeg is redistributed).

Produces plantseg_taxonomy_prompts.json, the file read by gdino_baseline_eval_plantseg_tax.py.
Each PlantSeg test image's disease name is mapped to the prompt the annotation pipeline
used for the same normalized label in final_gdino_results.jsonl; the raw name is used
when no entry exists, and the script prints the count (363 of 2,165 in the paper's run).

Input (work directory): plantseg_test_gt.jsonl (built by build_plantseg_test_gt.py from a
local copy of PlantSeg) and final_gdino_results.jsonl.
Output: plantseg_taxonomy_prompts.json, mapping coco_id to prompt.
final_gdino_results.jsonl is the output of construction step 3 (final_gdino_script.py) and records
the prompt the detector actually received for every label, which is why it is the source here.
Paths are relative to the repository root.
"""
import json


def main():
    prompt_by_label = {}
    for line in open("./work/final_gdino_results.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        lab = (r.get("stageD_label") or "").strip().lower()
        pr = r.get("stageE_prompt")
        if pr and lab and lab not in prompt_by_label:
            prompt_by_label[lab] = pr
    out, matched = {}, 0
    for l in open("./work/plantseg_test_gt.jsonl"):
        g = json.loads(l)
        lab = (g.get("disease_raw") or "").strip().lower()
        if lab in prompt_by_label:
            out[g["coco_id"]] = prompt_by_label[lab]
            matched += 1
        else:
            out[g["coco_id"]] = lab
    json.dump(out, open("./work/plantseg_taxonomy_prompts.json", "w"))
    print(f"PlantSeg test: taxonomy prompt found for {matched}/{len(out)} images; raw name used for the rest")


if __name__ == "__main__":
    main()
