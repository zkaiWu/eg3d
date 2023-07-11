#!/bin/bash

CIDX=$1

CUDA_VISIBLE_DEVICES=$CIDX python gen_samples_with_images_and_poses.py \
    --outdir=out \
    --exp_name=sample_direct_training_images_temp \
    --trunc=0.7 \
    --shapes=true \
    --seeds=0-3 \
    --camera_record_mode from_file \
    --camera_file /data5/wuzhongkai/proj/eg3d/eg3d/out/sample_generation/seed0001/meta.json \
    --network=/data5/wuzhongkai/proj/eg3d/eg3d/exp/training_directly/00000-ffhq-eg3d_generation_data_256floyd_ffhqformat-gpus2-batch32-gamma1/network-snapshot-002000.pkl