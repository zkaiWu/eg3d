import json
import numpy as np 
import argparse
import torch 


def parse_args():
    parser = argparse.ArgumentParser(description='Blender 64x64 to 256x256 using floyd')
    parser.add_argument('--output_dir', required=True, type=str, help='where to output the camera path json')
    parser.add_argument('--radius_range', nargs='+', type=float, default=[3.5, 3.5], help='radius range')
    # parser.add_argument('--pitch_range', nargs='+', type=float, default=[0.0, 0.0], help='pitch range')
    # parser.add_argument('--yaw_range', nargs='+', type=float, default=[0.0, 0.0], help='yaw range')
    parser.add_argument('--resolution', type=int, default=800, help='resolution')
    parser.add_argument('--sample_num', type=int, default=200, help='sample number')

    args = parser.parse_args()
    return args


def normalize_torch(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return x / (torch.norm(x, dim=dim, keepdim=True))


def normalize_np(x: np.ndarray, axis: int=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True))


def calc_camera_matrix(pitch, yaw, radius):

    camera_origin = np.array([
        radius * np.sin(yaw) * np.cos(pitch),
        radius * np.sin(yaw) * np.sin(pitch),
        radius * np.cos(yaw), 
    ])
    print(f"z: {camera_origin[2]}")

    forward_vector = -camera_origin 
    forward_vector = normalize_np(forward_vector)
    up_vector = np.array([0, 0, 1])

    right_vector = np.cross(forward_vector, up_vector)

    right_vector = normalize_np(right_vector)
    up_vector = np.cross(right_vector, forward_vector)
    up_vector = normalize_np(up_vector)

    camera_metrix = np.eye(4)
    camera_metrix[:3, :3] = np.stack([right_vector, up_vector, -forward_vector], axis=-1)
    camera_metrix[:3, 3] = camera_origin
    print(camera_metrix)
    
    return camera_metrix


def sample_pose_camera_path(args):
    camera_path = {}
    camera_path["seconds"] = 4             # fake seconds
    camera_path["camera_type"] = 'perspective'
    camera_path["render_height"] = camera_path["render_width"] = args.resolution
    camera_path["camera_path"] = []
    camera_path['keyframes'] = []
    camera_path['fps'] = 24
    camera_path['is_cycle'] = False
    camera_path['crop'] = None
    camera_path['smoothness_value'] = 0.5

    
    for i in range(args.sample_num):
        # pitch = np.random.uniform(args.yaw_range[0], args.yaw_range[1])
        # yaw = np.random.uniform(args.pitch_range[0], args.pitch_range[1])
        gamma_p = np.random.uniform(0, 2*np.pi)
        gamma_y = np.random.uniform(0, 0.5)
        pitch = gamma_p
        yaw = np.arccos(1 - 2 * gamma_y)
        radius = np.random.uniform(args.radius_range[0], args.radius_range[1])
        print(f'radius: {radius}')
        # print(pitch)
        # print(yaw)
        # print(radius)

        metrix = calc_camera_matrix(pitch, yaw, radius)
        meta_data = {
            'camera_to_world': metrix.reshape(16).tolist(),
            'fov': 50,
            'aspect': 1
        }
        camera_path["camera_path"].append(meta_data)
        matrix_str = custom_np_2_str(np.transpose(metrix).reshape(16))
        keyframe_data = {
            # 'matrix':  np.array2string(np.transpose(metrix).reshape(16), separator=',', prefix='[', suffix=']'),
            'matrix': matrix_str,
            'fov': 50,
            'aspect': 1,
            'properties': f'[["FOV",50],["NAME","Camera {i}"],["TIME",{i}]]',

        }
        camera_path['keyframes'].append(keyframe_data)
    
    with open(args.output_dir, 'w') as json_file:
        json.dump(camera_path, json_file, indent=4)



def custom_np_2_str(matrix: np.array):
    str = '['
    for i in matrix[:-1]:
        if i == 0:
            str += '0,'
        else:
            str += '{:.12f},'.format(i)
    str += '{:.12f}'.format(matrix[-1]) if matrix[-1] != 0 else '0'
    str += ']'

    return str


if __name__ == "__main__":
    args = parse_args()
    sample_pose_camera_path(args)
