#!/bin/bash

CIDX=$1
GPU_NUM=$2
BATCH_SIZE=$3

CUDA_VISIBLE_DEVICES=$CIDX python train.py \
    --outdir=exp/training_patch_debug \
    --cfg=ffhq \
    --data=/data5/wuzhongkai/data/dreamfusion_data/eg3d_generation_data_256floyd_ffhqformat \
    --gpus=$GPU_NUM \
    --batch=$BATCH_SIZE \
    --gamma=1 \
    --gen_pose_cond=True \
    --metrics fid2k_full_for_patch_rendering \
    --mbstd-group 2 \