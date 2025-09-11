import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.vis.camera_utils as camera_utils
import utils.jsonl_utils as jsonl_utils
import utils.mesh_utils as mesh_utils
import utils.vis.render_py3d as render_py3d
import utils.vis.render_cad as render_cad
import utils.img_utils as img_utils

import trimesh
from typing import Dict, Optional
import torch
from PIL import Image


def plot_normals(shape_path, output_path, cam_pos, see_at, normal_scale, device='cuda:0'):
    '''
    渲染base/sketch的noraml img，同时产生检查图像
    这里shape_path 需要是add的path
    '''
    add_input_path = shape_path  # /a/b/add/1.obj
    add_output_path = output_path  # /a/b/add/1.png

    add_input_parts = add_input_path.split(os.sep)
    add_output_parts = add_output_path.split(os.sep)

    # 所有输入路径
    add_input_parts[-2] = 'base'
    base_input_path = os.sep.join(add_input_parts)

    # 所有输出路径
    add_output_parts[-3] = 'base'
    base_output_path = os.sep.join(add_output_parts)
    add_output_parts[-3] = 'target'
    target_output_path = os.sep.join(add_output_parts)

    add_output_dir = os.path.dirname(add_output_path)
    base_output_dir = os.path.dirname(base_output_path)
    target_output_dir = os.path.dirname(target_output_path)

    os.makedirs(add_output_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(target_output_dir, exist_ok=True)

    tw_add = mesh_utils.TrimeshWrapper.load(add_input_path)
    mesh_add = tw_add.to_pytorch3d(device)
    tw_base = mesh_utils.TrimeshWrapper.load(base_input_path)
    mesh_base = tw_base.to_pytorch3d(device)

    plot_one_normal(mesh_add, add_output_path, device, cam_pos, see_at, scale=normal_scale)
    plot_one_normal(mesh_base, base_output_path, device, cam_pos, see_at, scale=normal_scale)

    add_img = Image.open(add_output_path)
    base_img = Image.open(base_output_path)

    target_img = img_utils.stack_imgs(base_img, add_img, mode='ew', output_path=target_output_path)


    return {'binary': target_img}


def plot_one_normal(mesh, output_path, device, cam_pos, see_at, scale):
    # tw = mesh_utils.TrimeshWrapper.load(mesh_path)
    # mesh = tw.to_pytorch3d(device)

    see_at_tensor = torch.tensor(see_at, dtype=torch.float32, device=device).unsqueeze(0)
    cam_pos_tensor = torch.tensor(cam_pos, dtype=torch.float32, device=device).unsqueeze(0)

    R, T = render_py3d.get_RT_from_cam(cam_pos_tensor, look_at=see_at_tensor)
    map = render_py3d.render_normal_map(mesh, device=device, image_size=512, R=R, T=T, scale=scale)
    render_py3d.save_png(map, output_path)


if __name__ == "__main__":
    cam_pos = [[2.0, 2.0, 2.0], [2.0, -2.0, 2.0], [2.0, 2.0, -2.0], [2.0, -2.0, -2.0]]
    cam_pos = [2.0, 2.0, -2.0]
    # plot_one_normal(device='cuda:0', cam_pos=cam_pos, see_at=[0.0, 0.0, 0.0])

    plot_normals(shape_path='/home/lkh/siga/dataset/ABC/new/c1/base_add/1/obj/add/00000006.obj', 
                 output_path='/home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/render/render_sketch/src/1/add/1.png', 
                 device='cuda:0', cam_pos=cam_pos, see_at=[0.0, 0.0, 0.0])