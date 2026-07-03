#!/bin/bash

set -Eeuo pipefail

# 创建输出目录
mkdir -p output_MLP/stage1

CUDA_VISIBLE_DEVICES=0 python train_MLP_modify.py \
        --train_csv "splits/dynamic_csvs/MGnik5.csv" \
        --out_dir "output_MLP/stage1" \
        --max_steps 15000 \
        --eval_steps 200 \
        --accum_steps 8 \
        --batch_size 1 \
        --margin 0.1 \
        --margin_weight 5.0 \
        --depth 2 \
        --rankH 128 \
        --mlp_hidden_dim 512 \
        --weight_decay 1e-3 \
        --lr 1e-4 \
        --num_workers 4 > "output_MLP/stage1/train_stage1_mgnify.log" 2>&1 &

wait
