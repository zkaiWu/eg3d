import PIL.Image
import shutil
import os
import argparse
import glob
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    return args


def custom_format2ffhq_format(args):
    image_dir = args.image_dir
    json_dict_ffhq = {}
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(args.output_path, exist_ok=True)
    json_dict_ffhq['labels'] = {}
    for tot_idx in os.listdir(image_dir):
        tot_idx_path = os.path.join(image_dir, tot_idx)
        image_paths = glob.glob(os.path.join(tot_idx_path, '*.png'))
        for img_p in image_paths:
            # shutil.copy(img_p, os.path.join(args.output_path, tot_idx + '_' + os.path.basename(img_p)))
            json_dict_ffhq['labels'][tot_idx + '_' + os.path.basename(img_p)] = data[int(tot_idx)]['camera_params'][0] 

    json_dict_ffhq['cam_pivot'] = data[0]['cam_pivot']
    json_dict_ffhq['cam_radius'] = data[0]['cam_radius']
        
    with open(os.path.join(args.output_path, 'dataset.json'), 'w') as f:
        json.dump(json_dict_ffhq, f, indent=4)
    


if __name__ == '__main__':
    args = parse_args()
    custom_format2ffhq_format(args)
    