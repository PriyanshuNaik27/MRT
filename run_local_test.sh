#!/bin/bash
python train/sft/scripts/sft_conversations.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name json \
    --dataset_train_split train \
    --train_file dummy_train.jsonl \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --dataset_num_proc 1 \
    --logging_steps 1 \
    --eval_strategy no \
    --output_dir ./test_output \
    --report_to none
