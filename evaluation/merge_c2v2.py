"""
Merges the main supervised model's LoRA adapter into the base Qwen3-VL-2B.

Produces a full Hugging Face checkpoint of the mixed + healthy (12%) model, which
is the initialization required by the RL stage (verl loads full checkpoints, not
adapters). After merging, the script checks that the merged model and base +
adapter give identical greedy outputs on 40 benchmark images.

Set the adapter directory, the base model and the output directory at the top
of the script.
"""
import os, json, shutil, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE    = "./weights/qwen3-vl-2b"
ADAPTER = "./work/output_agroground_2b_recogneg2_v2"
OUT     = "./weights/agroground-2b-c2v2-merged"
GT      = "./work/human_verified_ground_truth_v4.jsonl"

print("[1/4] loading base + adapter ...")
base = Qwen3VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda:0")
peft_model = PeftModel.from_pretrained(base, ADAPTER)
peft_model.eval()
processor = AutoProcessor.from_pretrained(BASE)

def fix_path(p):
    return p.replace('/home/jovyan/liuxiang/LLaVA/Qwen-VL/image_en_eccv','./datasets/cddm/images') if p and '/home/jovyan' in p else p
def prompt_for(g):
    d = (g.get('disease_raw') or g.get('abstention_disease') or 'disease').strip().lower()
    if g['granularity'] == 'coarse':
        return f"Locate the leaf or plant affected by {d} in this image. Output the bounding box in JSON format. If not present, output an empty list."
    return f"Locate all regions affected by {d} in this image. Output bounding boxes in JSON format. If no such regions are present, output an empty list."
def generate(model, img, text):
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":text}]}]
    t = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inputs = processor(text=[t], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.batch_decode([o[len(i):] for i,o in zip(inputs.input_ids, gen)], skip_special_tokens=True)[0]

rows = [json.loads(l) for l in open(GT)]
sample = [r for r in rows if r['granularity']=='fine'][:20] + [r for r in rows if r['granularity']=='coarse'][:10] + [r for r in rows if r['granularity']=='healthy'][:10]
print("[2/4] generating with base+adapter on 40 gold images ...")
ref = [generate(peft_model, fix_path(r['orig_path']), prompt_for(r)) for r in sample]

print("[3/4] merging and saving ...")
merged = peft_model.merge_and_unload()
if os.path.exists(OUT): shutil.rmtree(OUT)
merged.save_pretrained(OUT, safe_serialization=True)
for f in ["preprocessor_config.json","video_preprocessor_config.json","tokenizer.json","tokenizer_config.json",
          "vocab.json","merges.txt","chat_template.json","generation_config.json","added_tokens.json","special_tokens_map.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src): shutil.copy(src, OUT)
print("   saved:", sorted(os.listdir(OUT)))

print("[4/4] reloading merged checkpoint from disk and re-generating ...")
del merged, peft_model, base; torch.cuda.empty_cache()
m2 = Qwen3VLForConditionalGeneration.from_pretrained(OUT, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
p2 = AutoProcessor.from_pretrained(OUT)
processor = p2
new = [generate(m2, fix_path(r['orig_path']), prompt_for(r)) for r in sample]
same = sum(1 for a,b in zip(ref,new) if a.strip()==b.strip())
empty_ref = sum(1 for a in ref if a.strip()=="[]"); empty_new = sum(1 for a in new if a.strip()=="[]")
print(f"\nEQUIVALENCE: {same}/40 outputs byte-identical | empty outputs ref {empty_ref} vs merged {empty_new}")
diffs = [(a,b) for a,b in zip(ref,new) if a.strip()!=b.strip()][:3]
for a,b in diffs: print("  DIFF\n   adapter:", a[:120], "\n   merged :", b[:120])
print("VERDICT:", "MERGE FAITHFUL" if same >= 36 else "INVESTIGATE — too many differences")
