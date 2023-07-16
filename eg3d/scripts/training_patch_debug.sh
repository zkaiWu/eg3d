#!/bin/bash

CIDX=$1
GPU_NUM=$2
BATCH_SIZE=$3
MBSTD=$4
OUTPUT_DIR=$5

CUDA_VISIBLE_DEVICES=$CIDX python train.py \
    --outdir=$OUTPUT_DIR \
    --cfg=ffhq \
    --data=/home/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_256floyd_ffhqformat \
    --gpus=$GPU_NUM \
    --batch=$BATCH_SIZE \
    --gamma=1 \
    --gen_pose_cond=True \
    --mbstd-group $MBSTD \