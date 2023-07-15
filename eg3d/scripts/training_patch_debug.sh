#!/bin/bash

CIDX=$1
GPU_NUM=$2
BATCH_SIZE=$3
OUTPUT_DIR=$4

CUDA_VISIBLE_DEVICES=$CIDX python train.py \
    --outdir=$OUTPUT_DIR \
    --cfg=ffhq \
    --data=/home2/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise250_promptlist \
    --resume=/home/zhongkaiwu/proj/eg3d/eg3d/exp/training_patch_debug/00003-ffhq-eg3d_generation_data_g4-gpus4-batch16-gamma1/network-snapshot-002800.pkl \
    --gpus=$GPU_NUM \
    --batch=$BATCH_SIZE \
    --gamma=1 \
    --gen_pose_cond=True \
    --metrics none \
    --mbstd-group $GPU_NUM \