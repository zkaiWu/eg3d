import os
import sys
import PIL.Image as Image
import glob
import json
import copy
import argparse
import shutil
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
import glob


def lr_process(args):
    input_dir = args.input_dir 
    resolution = args.resolution

    for obj_name in os.listdir(input_dir):
        obj_dir = os.path.join(input_dir, obj_name)

        image_dirs = os.path.join(obj_dir, 'images')
        transforms_path = os.path.join(obj_dir, 'meta.json')
        image_output_dir = os.path.join(obj_dir, 'images_{}'.format(args.output_suffix))
        os.makedirs(image_output_dir, exist_ok=True)

        # process image
        print(image_dirs)
        image_path_list = glob.glob(os.path.join(image_dirs, '*.png')) 
        print(image_path_list)
        for img_path in image_path_list:
            img = Image.open(img_path)
            img_lr = img.resize((resolution, resolution), Image.BICUBIC)
            img_lr.save(os.path.join(image_output_dir, os.path.basename(img_path)))

        # process json
        try:
            with open(transforms_path, 'r') as f:
                json_data = json.load(f)
        except:
            print("not found json_data")
            continue

        json_output_path = os.path.join(obj_dir, 'meta_{}.json'.format(args.output_suffix))
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=4) 
        # for f_idx, frame in enumerate(json_data['frames']):
        #     print(frame)
        #     # size = min(frame['w'], frame['h'])
        #     file_path = os.path.join('.', 'images_{}'.format(args.output_suffix), os.path.basename(frame['file_path']))
        #     w = args.resolution
        #     h = args.resolution

        #     # assign
        #     scale_h = frame['h'] / h
        #     scale_w = frame['w'] / w
        #     json_data['frames'][f_idx]['w'] = w
        #     json_data['frames'][f_idx]['h'] = h
        #     json_data['frames'][f_idx]['cx'] = frame['cx'] / scale_w
        #     json_data['frames'][f_idx]['cy'] = frame['cy'] / scale_h
        #     json_data['frames'][f_idx]['fl_x'] = frame['fl_x'] / scale_w
        #     json_data['frames'][f_idx]['fl_y'] = frame['fl_y'] / scale_h 
        #     json_data['frames'][f_idx]['file_path'] = file_path


            
def parse_args():
    parser = argparse.ArgumentParser(description='Blender LR image processor')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--output_suffix', required=True, type=str, help='output image directory')
    parser.add_argument('--resolution', default=64, type=int, help='target resolution')
    args = parser.parse_args()
    return args


"""
python dataset_preprocessing/sr3d_data_generation/downsample_data_generation.py --input_dir /data5/wuzhongkai/data/dreamfusion_data/eg3d_generation_data --output_suffix 64x64 --resolution 64
"""

if __name__ == '__main__':

    args = parse_args()
    lr_process(args)