# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""
"""In this scripts we also save pose and images"""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import json


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def create_cam2world_angle(sample_num, angle_p_range, angle_y_range):
    angle_p_sample_list = np.random.uniform(angle_p_range[0], angle_p_range[1], sample_num)
    angle_y_sample_list = np.random.uniform(angle_y_range[0], angle_y_range[1], sample_num)
    
    return list(zip(angle_y_sample_list, angle_p_sample_list))





@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--sample_num', help='pose and images number sampled', type=int, required=False, metavar='int', default=200, show_default=True)
@click.option('--exp_name', help='exp_name will be used as output dir as ./out/{exp_name}', type=str, required=True, metavar='str', default='exp', show_default=True)
@click.option('--camera_record_mode', help='random sample or read from file, choose from [random_sample, from_file]', type=str, required=False, metavar='str', default='random_sample', show_default=True)
@click.option('--camera_file', help='camera parameters file', type=str, required=False, metavar='str', default='meta.json', show_default=True)
@click.option('--angle_p_range', help='range of angle_p', type=parse_vec2, required=False, metavar='float,float', default=(-0.5, 0.5), show_default=True)
@click.option('--angle_y_range', help='range of angle_y', type=parse_vec2, required=False, metavar='float,float', default=(-0.5, 0.5), show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    sample_num: int,
    exp_name: str,
    camera_record_mode: str,
    camera_file: str,
    angle_p_range: Tuple[float, float],
    angle_y_range: Tuple[float, float]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        imgs = []
        meta_json = []
        if camera_record_mode == 'random_sample':
            angle_list = create_cam2world_angle(sample_num, angle_p_range, angle_y_range)
            for angle_y, angle_p in tqdm(angle_list):
            # angle_p = -0.0
            # for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, camera_params)['image']

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

                meta_data = {
                    'cam_pivot': cam_pivot.cpu().numpy().tolist(),
                    'cam_radius': cam_radius,
                    'camera_params': camera_params.cpu().numpy().tolist(),
                    'conditioning_params': conditioning_params.cpu().numpy().tolist(),
                }

                meta_json.append(meta_data)
        
        elif camera_record_mode == 'from_file':
            # read camera parameters from json file 
            meta_json = None
            if camera_file.endswith('.json'):
                with open(camera_file, 'r') as f:
                    meta_json = json.load(f)
            
            assert meta_json is not None, f'camera_file {camera_file} does not exist'
            for i in tqdm(range(len(meta_json))):
            # angle_p = -0.0
            # for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                # cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                # cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                # conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                # camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                # conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                meta_data = meta_json[i]
                cam_pivot = torch.tensor(meta_data['cam_pivot'], device=device)
                cam_radius = meta_data['cam_radius']
                camera_params = torch.tensor(meta_data['camera_params'], device=device)
                conditioning_params = torch.tensor(meta_data['conditioning_params'], device=device)

                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, camera_params)['image']

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

                meta_data = {
                    'cam_pivot': cam_pivot.cpu().numpy().tolist(),
                    'cam_radius': cam_radius,
                    'camera_params': camera_params.cpu().numpy().tolist(),
                    'conditioning_params': conditioning_params.cpu().numpy().tolist(),
                }

                meta_json.append(meta_data)
                

        print('aaaaa')
        output_dir = os.path.join(outdir, exp_name, f'seed{seed:04d}')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        for i, img in enumerate(imgs):
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(output_dir, 'images', f'{i}.png'))
        
        with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
            json.dump(meta_json, f, indent=4)

        # if shapes:
        #     # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        #     max_batch=1000000

        #     samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        #     samples = samples.to(z.device)
        #     sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        #     transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        #     transformed_ray_directions_expanded[..., -1] = -1

        #     head = 0
        #     with tqdm(total = samples.shape[1]) as pbar:
        #         with torch.no_grad():
        #             while head < samples.shape[1]:
        #                 torch.manual_seed(0)
        #                 sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
        #                 sigmas[:, head:head+max_batch] = sigma
        #                 head += max_batch
        #                 pbar.update(max_batch)

        #     sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        #     sigmas = np.flip(sigmas, 0)

        #     # Trim the border of the extracted cube
        #     pad = int(30 * shape_res / 256)
        #     pad_value = -1000
        #     sigmas[:pad] = pad_value
        #     sigmas[-pad:] = pad_value
        #     sigmas[:, :pad] = pad_value
        #     sigmas[:, -pad:] = pad_value
        #     sigmas[:, :, :pad] = pad_value
        #     sigmas[:, :, -pad:] = pad_value

        #     if shape_format == '.ply':
        #         from shape_utils import convert_sdf_samples_to_ply
        #         convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
        #     elif shape_format == '.mrc': # output mrc
        #         with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        #             mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
