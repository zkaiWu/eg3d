#!/bin/bash

CIDX=$1
GPU_NUM=$2
BATCH_SIZE=$3
OUTDIR=$4

CUDA_VISIBLE_DEVICES=$CIDX python train.py \
    --outdir=$OUTDIR \
    --cfg=ffhq \
    --data=/home2/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise500_long_prompt \
    --gpus=$GPU_NUM \
    --batch=$BATCH_SIZE \
    --gamma=1 \
    --gen_pose_cond=True \
    --kimg 10000 \
    --use_mimic3d true \
    --resume /home/zhongkaiwu/proj/eg3d/eg3d/exp/training_directly_noise500_long_prompt/00000-ffhq-eg3d_generation_data_g4-gpus8-batch64-gamma1/network-snapshot-006451.pkl \
    --resume_from eg3d \


