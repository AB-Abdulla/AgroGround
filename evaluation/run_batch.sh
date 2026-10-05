#!/bin/bash
# Full evaluation batch on the benchmark and on PlantSeg: every model and protocol reported in
# the paper, the reasoning baselines, the open-set runs. Produces the prediction files scored
# by score.py, with the exact file names listed in the README's reproduction map.
# usage, from the repository root: bash evaluation/run_batch.sh <GPU_A> <GPU_B>
#   (two GPUs; about 12 hours; every script is resumable). Controls: evaluation/run_controls.sh.
#
# Scripts read ./work (intermediate files, the benchmark, the PlantSeg ground truth built by
# evaluation/build_plantseg_test_gt.py) and ./weights. Adapters are the LoRA output directories
# of the SFT launchers in configs/sft; RL checkpoints are the merged full checkpoints of the GRPO
# runs in configs/rl; the 7B baselines and the base models are their public releases.
A=$1; B=$2
W=./weights                                             # qwen3-vl-2b, qwen3-vl-8b, VisionReasoner-7B, Seg-Zero-7B, RL checkpoints
AD_A=./work/output_agroground_2b                        # grounding only
AD_B2=./work/output_agroground_2b_recog_v2              # mixed
AD_C3=./work/output_agroground_2b_recogneg3_v2          # mixed + healthy (6%)
AD_C2=./work/output_agroground_2b_recogneg2_v2          # mixed + healthy (12%), the main supervised model
AD_S1=./work/output_agroground_2b_recogneg2_v2_seed1    # seed study
AD_S2=./work/output_agroground_2b_recogneg2_v2_seed2
AD_8B=./work/output_agroground_8b_recogneg2             # 8B model
AD_PS=./work/output_plantseg_2b                         # expert-label baseline (PlantSeg)
AD_PEQ=./work/output_pseudo_eq_2b                       # equal-size pseudo-label run
AD_10K=./work/output_sub10k_2b                          # scaling subsets
AD_100K=./work/output_sub100k_2b
RL170=$W/agroground-2b-grpo-s170                        # main RL model (step 170)
RL292=$W/agroground-2b-grpo-s292                        # final checkpoint (step 292)
RLB=$W/agroground-2b-grpoB-s146                         # recognition-weighted reward ablation
RLC=$W/agroground-2b-grpoC-s146                         # uniform rollout sampling ablation
E=evaluation

# --- GPU A: GroundingDINO baselines, known-target protocol on the benchmark, reasoning baselines ---
(
CUDA_VISIBLE_DEVICES=$A python3 -u $E/gdino_baseline_eval_v4.py
CUDA_VISIBLE_DEVICES=$A python3 -u $E/gdino_baseline_eval_tax_v4.py
CUDA_VISIBLE_DEVICES=$A python3 -u $E/gdino_recognition_baseline_v4.py
python3 -u $E/eval_qwen_v4.py --adapter none                       --protocol given --gpu $A --mem_fraction 0.5 --out qwen_ZERO_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $W/qwen3-vl-8b --protocol given --gpu $A --mem_fraction 0.7 --out qwen_ZERO8B_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_A                       --protocol given --gpu $A --mem_fraction 0.5 --out qwen_A_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_B2                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_B2_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_C3                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_C3v2_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_C2                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_C2v2_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_S1                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_C2s1_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_S2                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_C2s2_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_8B --model $W/qwen3-vl-8b --protocol given --gpu $A --mem_fraction 0.7 --out qwen_SFT8B_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_PS                      --protocol given --gpu $A --mem_fraction 0.5 --out qwen_PS_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_PEQ                     --protocol given --gpu $A --mem_fraction 0.5 --out qwen_PEQ_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_10K                     --protocol given --gpu $A --mem_fraction 0.5 --out qwen_S10K_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_100K                    --protocol given --gpu $A --mem_fraction 0.5 --out qwen_S100K_given_v4.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RL170         --protocol given --gpu $A --mem_fraction 0.5 --out qwen_RL170_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RL292         --protocol given --gpu $A --mem_fraction 0.5 --out qwen_RLfinal_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RLB           --protocol given --gpu $A --mem_fraction 0.5 --out qwen_RLB_given_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RLC           --protocol given --gpu $A --mem_fraction 0.5 --out qwen_RLC_given_v4.json
python3 -u $E/eval_reasoner_v4.py --model_path $W/VisionReasoner-7B --mode multi  --protocol given --gpu $A --mem_fraction 0.7 --out vr_given_v3.json
python3 -u $E/eval_reasoner_v4.py --model_path $W/Seg-Zero-7B       --mode single --protocol given --gpu $A --mem_fraction 0.7 --out segzero_given_v3.json
echo BATCH_A_DONE
) &

# --- GPU B: candidate protocol, open-set runs, PlantSeg transfer ---
(
python3 -u $E/eval_qwen_v4.py --adapter none                       --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_ZERO_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $W/qwen3-vl-8b --protocol recognition --gpu $B --mem_fraction 0.7 --out qwen_ZERO8B_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_A                       --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_A_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_B2                      --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_B2_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_C3                      --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_C3v2_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_C2                      --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_C2v2_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_S1                      --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_C2s1_recog_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_S2                      --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_C2s2_recog_v4.json
python3 -u $E/eval_qwen_v4.py --adapter $AD_8B --model $W/qwen3-vl-8b --protocol recognition --gpu $B --mem_fraction 0.7 --out qwen_SFT8B_recog_v4.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RL170         --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_RL170_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RL292         --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_RLfinal_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RLB           --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_RLB_recog_v3.json
python3 -u $E/eval_qwen_v4.py --adapter none --model $RLC           --protocol recognition --gpu $B --mem_fraction 0.5 --out qwen_RLC_recog_v4.json
python3 -u $E/eval_reasoner_v4.py --model_path $W/VisionReasoner-7B --mode multi --protocol recognition --gpu $B --mem_fraction 0.7 --out vr_recog_v3.json
python3 -u $E/eval_openset_v4.py --adapter none    --gpu $B --mem_fraction 0.5 --out openset2_ZERO_v4.json
python3 -u $E/eval_openset_v4.py --adapter $AD_A   --gpu $B --mem_fraction 0.5 --out openset2_A_v4.json
python3 -u $E/eval_openset_v4.py --adapter $AD_B2  --gpu $B --mem_fraction 0.5 --out openset2_B2_v4.json
python3 -u $E/eval_openset_v4.py --adapter $AD_C2  --gpu $B --mem_fraction 0.5 --out openset2_C2v2_v4.json
python3 -u $E/eval_openset_v4.py --adapter none --model $RL170 --gpu $B --mem_fraction 0.5 --out openset2_RL170_v4.json
python3 -u $E/eval_openset_v4.py --adapter none --model $RLB   --gpu $B --mem_fraction 0.5 --out openset2_RLB_v4.json
python3 -u $E/eval_openset_v4.py --adapter none --model $RLC   --gpu $B --mem_fraction 0.5 --out openset2_RLC_v4.json
python3 -u $E/eval_openset_v3.py --adapter none    --gpu $B --out openset_ZERO_v4.json      # first prompt formulation
python3 -u $E/eval_openset_v3.py --adapter $AD_A   --gpu $B --out openset_A_v4.json
python3 -u $E/eval_openset_v3.py --adapter $AD_B2  --gpu $B --out openset_B2_v4.json
python3 -u $E/eval_openset_v3.py --adapter $AD_C2  --gpu $B --out openset_C2v2_v4.json
# PlantSeg transfer (requires ./work/plantseg_test_gt.jsonl and plantseg_taxonomy_prompts.json, both built locally)
CUDA_VISIBLE_DEVICES=$B python3 -u $E/gdino_baseline_eval_plantseg.py
CUDA_VISIBLE_DEVICES=$B python3 -u $E/gdino_baseline_eval_plantseg_tax.py
python3 -u $E/eval_qwen_plantseg.py --adapter none                       --gpu $B --mem_fraction 0.5 --out qwen_ZERO_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter none --model $W/qwen3-vl-8b --gpu $B --mem_fraction 0.7 --out qwen_ZERO8B_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter $AD_A                       --gpu $B --mem_fraction 0.5 --out qwen_A_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter $AD_C2                      --gpu $B --mem_fraction 0.5 --out qwen_C2v2_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter none --model $RL170         --gpu $B --mem_fraction 0.5 --out qwen_RL170_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter $AD_8B --model $W/qwen3-vl-8b --gpu $B --mem_fraction 0.7 --out qwen_SFT8B_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter $AD_PS                      --gpu $B --mem_fraction 0.5 --out qwen_PS_given_ps.json
python3 -u $E/eval_qwen_plantseg.py --adapter $AD_PEQ                     --gpu $B --mem_fraction 0.5 --out qwen_PEQ_given_ps.json
echo BATCH_B_DONE
) &
wait
