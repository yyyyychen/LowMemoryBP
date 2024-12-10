#!/bin/bash

TASK=stsb
SEED=1

CUDA_VISIBLE_DEVICES=0,1 python run_glue.py \
    --model_name_or_path ./ckpt/roberta-base-mergeln \
    --task_name ${TASK} \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate 0.0005 \
    --num_train_epochs 30 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --output_dir ./output \
    --seed $SEED \
    --save_total_limit 1 \
    --save_steps 1000 \
    --lora_r 64 \
    --lora_alpha 16 \
    --disable_tqdm False \
    --skip_memory_metrics False \
    --overwrite_output_dir \
    --using_method activation_and_mergeln \
    --data_path ./datasets/glue \
