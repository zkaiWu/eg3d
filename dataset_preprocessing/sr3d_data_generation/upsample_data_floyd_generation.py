import torch
import torch.nn as nn
import os
import sys
from PIL import Image, ImageOps
import glob
import json
import copy
import argparse
import shutil
import torch
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers import IFSuperResolutionPipeline, IFPipeline
from diffusers.utils import pt_to_pil
from torch.utils.data import Dataset, DataLoader
import glob
import os
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import T5Tokenizer, T5EncoderModel




class eg3dDataset(Dataset):
    def __init__(self, input_dir) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.get_mate_data()

    def get_mate_data(self):
        self.image_paths = []
        self.obj_names = os.listdir(self.input_dir)
        self.mate_data = []
        for obj_name in self.obj_names:
            image_paths = glob.glob(os.path.join(self.input_dir, obj_name, 'images', '*.png'))
            image_paths.sort()
            for i, img_p in enumerate(image_paths):

                data_dict = {
                    'image_path': img_p, 
                    'obj_name': obj_name,
                    'idx': i,
                }
                self.mate_data.append(data_dict)

    
    def __len__(self):
        return len(self.mate_data)
    
    def __getitem__(self, index):
        img = Image.open(self.mate_data[index]['image_path']).convert('RGB')
        # white_background = Image.new('RGBA', img.size, (255, 255, 255, 25))
        # white_background.paste(img, mask=img.split()[3]) 
        # img = white_background.convert('RGB')
        img = transforms.Resize((64, 64), Image.BICUBIC)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        data_dict = self.mate_data[index]
        data_dict['image'] = img
        return data_dict
        


def image_sr(rank, world_size, args):

    print(f"rank is {rank}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')

    stage_1_id = "DeepFloyd/IF-I-XL-v1.0"
    stage_2_id = "DeepFloyd/IF-II-L-v1.0"
    # tokenizer = DiffusionPipeline.from_pretrained(stage_1_id, subfolder='tokenizer', cache_dir='/data5/wuzhongkai/diffusers_models', torch_dtype=torch.float16, local_files_only=True)
    # encoder = DiffusionPipeline.from_pretrained(stage_1_id, subfolder='text_encoder', cache_dir='/data5/wuzhongkai/diffusers_models', torch_dtype=torch.float16, local_files_only=True)
    # stage_1 = IFPipeline(tokenizer=tokenizer, text_encoder=encoder).to(rank)
    stage_1 = IFPipeline.from_pretrained(stage_1_id, variant="fp16", torch_dtype=torch.float16, cache_dir='/data5/wuzhongkai/diffusers_models', local_files_only=True).to(rank)
    stage_1.enable_model_cpu_offload(gpu_id=rank)
    stage_2 = DiffusionPipeline.from_pretrained(stage_2_id, text_encoder=None, variant="fp16", torch_dtype=torch.float16, cache_dir='/data5/wuzhongkai/diffusers_models', local_files_only=True).to(rank)
    stage_2.enable_model_cpu_offload(gpu_id=rank)

    prompt = ""
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt) 
    del stage_1
    

    input_dir = args.input_dir 
    output_dir = args.output_dir

    train_dataset = eg3dDataset(input_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4)

    dataloader_list = [
        train_dataloader,
        # test_dataloader,
        # val_dataloader,
    ]


    for dataloader in dataloader_list:
        for data in dataloader:
            image = data['image']
            image_path = data['image_path']
            obj_name = data['obj_name']
            prompt_embeds_batch = prompt_embeds.repeat(len(image), 1, 1)
            negative_embeds_batch = negative_embeds.repeat(len(image), 1, 1)
            for per_view_idx in range(args.image_per_view):
                upscaled_image = stage_2(
                    image=image, prompt_embeds=prompt_embeds_batch, negative_prompt_embeds=negative_embeds_batch,  output_type="pt",
                    guidance_scale=args.guidance_scale, noise_level=args.noise_level, num_inference_steps=args.num_inference_steps
                ).images
                for i in range(upscaled_image.shape[0]):
                    single_upscaled_image = pt_to_pil(upscaled_image)[i]
                    save_dir = os.path.join(output_dir, obj_name[i], f'images_{args.output_suffix}', f'{os.path.basename(image_path[i]).rstrip(".png")}')
                    os.makedirs(save_dir, exist_ok=True)
                    single_upscaled_image.save(os.path.join(save_dir, f'{per_view_idx}.png'))


def copy_json(args):

    input_dir = args.input_dir 
    output_dir = args.output_dir

    for obj_name in os.listdir(input_dir):
        output_obj_name = obj_name 
        obj_dir = os.path.join(input_dir, obj_name)
        if not os.path.isdir(obj_dir):
            continue

        for sub_name in os.listdir(obj_dir):
            sub_name = os.path.join(obj_dir, sub_name)

            if sub_name.endswith('.json'):
                json_path = sub_name
                os.makedirs(os.path.join(output_dir, output_obj_name), exist_ok=True)
                shutil.copy(json_path, os.path.join(output_dir, output_obj_name, f'meta_{args.output_suffix}.json'))



def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--input_dir', required=True, type=str, help='input image directory')
    parser.add_argument('--output_dir', required=True, type=str, help='output image directory')
    parser.add_argument('--output_suffix', required=True, type=str, help='output suffix')
    parser.add_argument('--guidance_scale', default=0.0, type=float, help='guidance scale for floyd')
    parser.add_argument('--noise_level', default=0, type=int, help='noise level for floyd')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='num inference steps for floyd')
    parser.add_argument('--image_per_view', type=int, default=50, help='number of images per view')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--world_size', type=int, default=4, help='world size use in ddp')
    parser.add_argument('--port', type=int, default=5678, help='world size use in ddp')

    args = parser.parse_args()
    return args


def main(args):
    world_size = args.world_size
    print(world_size)
    # copy_json(args)
    mp.spawn(image_sr, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    
    args = parse_args()
    main(args)