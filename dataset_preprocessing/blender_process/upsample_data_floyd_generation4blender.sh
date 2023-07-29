#!/bin/bash


CIDX=$1
WS=$2 # world size

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name lego \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/lego \

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name chair \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/chair \

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name drums \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/drums \

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name ficus \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/ficus \

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name hotdog \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/hotdog \

CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name materials \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/materials \


CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name mic \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/mic \


CUDA_VISIBLE_DEVICES=$CIDX python dataset_preprocessing/blender_process/upsample_data_floyd_generation4blender.py \
    --input_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5 \
    --batch_size 128 \
    --world_size $WS \
    --port 4567 \
    --obj_name ship \
    --output_dir /home2/zhongkaiwu/data/dreamfusion_data/blender4eg3d_r3.5_floyd_sr/ship \