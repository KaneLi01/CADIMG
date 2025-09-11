import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_utils import TrimeshWrapper

import numpy as np
import torch
from PIL import Image

from pytorch3d.renderer import (
    look_at_view_transform, OrthographicCameras, RasterizationSettings, MeshRasterizer, BlendParams, TexturesVertex, MeshRenderer, HardFlatShader
)


def get_RT_from_cam(cam_pos, look_at):
    R, T = look_at_view_transform(eye=cam_pos, at=look_at, up=((0, 0, 1),))
    return R, T

def render_normal_map(mesh, device, image_size, R, T, scale):

    cameras = OrthographicCameras(device=device, R=R, T=T, focal_length=scale)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    face_idx = fragments.pix_to_face[0, ..., 0]  # (H, W)
    valid_mask = face_idx >= 0

    # Object-space face normals
    face_normals_obj = mesh.faces_normals_packed()  # (F, 3)

    # 转换为 camera space normals
    R_cam = cameras.R[0].to(device)  # (3, 3)
    R_cam[..., :, 2] *= -1
    R_cam[..., :, 0] *= -1
    face_normals_cam = torch.matmul(R_cam.T, face_normals_obj.T).T  # (F, 3)

    # 映射到 [0, 1] 作为 RGB
    # face_normals_cam[:, 2] *= -1  # B取反 
    face_normals_rgb = (face_normals_cam + 1.0) / 2.0  # (F, 3)

    # 构建图像
    H, W = face_idx.shape
    normal_map = torch.ones((H, W, 3), device=device)
    normal_map[valid_mask] = face_normals_rgb[face_idx[valid_mask]]

    return normal_map.cpu().numpy()


def render_dot_map(mesh, device, image_size, R, T):
    # 设置相机
    cameras = OrthographicCameras(device=device, R=R, T=T, focal_length=0.9)
    
    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
    )
    
    # 创建渲染器 - 不使用光照，设置背景为白色
    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))  # 白色背景
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params
        )
    )
    
    verts = mesh.verts_packed()
    if not hasattr(mesh, 'textures') or mesh.textures is None:
        black_color = torch.zeros_like(verts).unsqueeze(0)  # 创建全黑颜色
        mesh.textures = TexturesVertex(verts_features=black_color)
    
    # 渲染mesh
    images = renderer(mesh)

    return images[0, ..., :3].cpu().numpy()   


def save_png(map, output_path):
    map_uint8 = (map * 255).astype(np.uint8)
    Image.fromarray(map_uint8).save(output_path)


def load_mesh(mesh_path, device, eye, at):
    """加载mesh并初始化"""
    mesh_tm = TrimeshWrapper.load(mesh_path)
    mesh_tm.normalization()
    mesh_tm.fix_face_orientation()    
    mesh_tm.save('/home/lkh/siga/output/temp/nor/0126_.ply')

    eye_tensor = torch.tensor([eye], dtype=torch.float32, device=device)
    at_tensor = torch.tensor([at], dtype=torch.float32, device=device)
    R, T = look_at_view_transform(eye=eye_tensor, at=at_tensor, up=((0, 0, 1),))
    mesh = mesh_tm.to_pytorch3d(flip=False)
    return mesh, R, T


def main():
    mesh_path = '/home/lkh/siga/dataset/ABC/temp/obj/dot/00/00000029/00000029_ad34a3f60c4a4caa99646600_step_009.obj'
    output_path = '/home/lkh/siga/output/temp/3d_1.png'
    image_size = 512
    cam_pos = np.array([2,2,2])
    see_at = np.array([0,0,0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh, R, T = load_mesh(mesh_path, device=device, eye=cam_pos, at=see_at)
    # normal_map = render_normal_map(mesh.to(device), device, image_size, R, T)
    map = render_dot_map(mesh.to(device), device, image_size, R, T)

    map_uint8 = (map * 255).astype(np.uint8)
    Image.fromarray(map_uint8).save(output_path)


if __name__ == "__main__":
    main()