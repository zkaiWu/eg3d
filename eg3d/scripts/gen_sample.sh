#!/bin/bash

CIDX=$1

CUDA_VISIBLE_DEVICES=$CIDX python gen_samples.py \
    --outdir=out/sample_direct_training_unseen_poes \
    --trunc=1.0 \
    --shapes=true \
    --seeds=0-3 \
    --network=/data5/wuzhongkai/proj/eg3d/eg3d/exp/training_directly/00000-ffhq-eg3d_generation_data_256floyd_ffhqformat-gpus2-batch32-gamma1/network-snapshot-002000.pkl