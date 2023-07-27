
import torch
import math 
import argparse
import json
import os

def cal_radius(matrix):
    origin = matrix[:3, 3]
    origin_norm = torch.norm(origin, dim=-1)
    return origin_norm
    

def cal_radius_of_blender(args):
    input_dir = args.input_dir
    obj_name = args.obj_name
    json_path = os.path.join(input_dir, obj_name, 'transforms_test.json')
    with open(json_path, 'r') as jfp:
        data = json.load(jfp)

    frame_datas = data['frames']
    for frame in frame_datas:
        camera_matrix = frame['transform_matrix']
        radius = cal_radius(torch.tensor(camera_matrix))
        print(f"{radius:.12f}")

    

def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--obj_name', type=str, help='which obj to sr')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cal_radius_of_blender(args)
    

    
