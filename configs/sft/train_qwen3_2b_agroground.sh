#!/bin/bash
# AgroGround SFT launcher: Qwen3-VL-2B + LoRA, grounding-only data (known-target instructions only).
# Adapted from the official qwen-vl-finetune scripts/sft_qwen3_4b.sh. Two GPUs.
# Set the GPU ids in the CUDA_VISIBLE_DEVICES line at the bottom.
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=2                       # 2 GPUs

deepspeed=./scripts/zero2.json         # ZeRO-2, good for LoRA
llm=./weights/qwen3-vl-2b

lr=1e-4                                 # standard LoRA LR
batch_size=4
grad_accum_steps=4                      # effective batch = 4 * 4 * 2 GPUs = 32

entry_file=qwenvl/train/train_qwen.py
datasets=agroground_train
run_name="qwen3vl_2b_agroground_ep1"
output_dir=./work/output_agroground_2b

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

cd ./Qwen3-VL/qwen-vl-finetune
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
