from typing import Tuple, Dict, Callable, Any, List
import torch
import torch.nn.functional as F
import numpy as np


def sample_patch_params(batch_size:int, patch_cfg, device:str='cuda'):

    scale = patch_cfg['patch_res'] / patch_cfg['hr_res']
    offsets = torch.rand(batch_size, 2, device=device) * (1 - scale)

    return {'scales': torch.tensor([[scale, scale]]).repeat(batch_size, 1).to(device), 'offsets': offsets}
    


def extract_patches(img, patch_params:Dict=None, resolution=64):
    """
    Extracts patches from images and interpolates them to a desired resolution
    Assumes, that scales/offests in patch_params are given for the [0, 1] image range (i.e. not [-1, 1])
    """
    _, _, h, w = img.shape
    assert h == w, "Can only work on square images (for now)"
    coords = compute_patch_coords(patch_params, resolution) # [batch_size, resolution, resolution, 2]
    out = F.grid_sample(img, coords, mode='bilinear', align_corners=True) # [batch_size, c, resolution, resolution]
    return out


def compute_patch_coords(patch_params: Dict, resolution: int, align_corners: bool=True, for_grid_sample: bool=True) -> torch.Tensor:
    """
    Given patch parameters and the target resolution, it extracts
    """
    patch_scales, patch_offsets = patch_params['scales'], patch_params['offsets'] # [batch_size, 2], [batch_size, 2]
    batch_size, _ = patch_scales.shape
    coords = generate_coords(batch_size=batch_size, img_size=resolution, device=patch_scales.device, align_corners=align_corners) # [batch_size, out_h, out_w, 2]

    # First, shift the coordinates from the [-1, 1] range into [0, 2]
    # Then, multiply by the patch scales
    # After that, shift back to [-1, 1]
    # Finally, apply the offset converted from [0, 1] to [0, 2]
    coords = (coords + 1.0) * patch_scales.view(batch_size, 1, 1, 2) - 1.0 + patch_offsets.view(batch_size, 1, 1, 2) * 2.0 # [batch_size, out_h, out_w, 2]

    # if for_grid_sample:
    #     # Transforming the coords to the layout of `F.grid_sample`
    #     coords[:, :, :, 1] = -coords[:, :, :, 1] # [batch_size, out_h, out_w]

    return coords


def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> torch.Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[idx, 0, 0] = (-1, 1)
    # - lower right corner: coords[idx, -1, -1] = (1, -1)
    # In this way, the `y` axis is flipped to follow image memory layout
    - lower right corner: coords[idx, -1, -1] = (-1, 1) 
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]
    coords = coords.permute(0, 2, 3, 1) # [batch_size, 2, img_size, img_size]

    return coords


    