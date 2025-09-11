
import blenderproc as bproc
import numpy as np
import trimesh
import pickle
import os
import argparse
from math import sqrt
import json


VIEW_CORNERS = [
    [ 2.0, -2.0,  2.0],
    [ 2.0, -2.0, -2.0],
    [ 2.0,  2.0, -2.0],
    [ 2.0,  2.0,  2.0],
    [-2.0, -2.0,  2.0],
    [-2.0, -2.0, -2.0],
    [-2.0,  2.0, -2.0],
    [-2.0,  2.0,  2.0],
]

VIEW_CORNERS_6 = {
    'front': [ # front
        [2.0,-2*sqrt(2),0.0], [2.0,-2.0,2.0], [2.0,0.0,2*sqrt(2)], [2.0,2.0,2.0], [2.0,2*sqrt(2),0.0], [2.0,0.0,-2*sqrt(2)]  
    ],

    'back': [ # back
        [-2.0,-2*sqrt(2),0.0], [-2.0,-2.0,2.0], [-2.0,0.0,2*sqrt(2)], [-2.0,2.0,2.0], [-2.0,2*sqrt(2),0.0], [-2.0,0.0,-2*sqrt(2)]
    ],

    'right': [ # right
        [0.0,2.0,2*sqrt(2)], [2.0,2.0,2.0], [2*sqrt(2),2.0,0.0], [2.0,2.0,-2.0], [0.0,2.0,-2*sqrt(2)], [-2*sqrt(2),2.0,0.0]
    ],
    
    'left': [ # left
        [0.0,-2.0,2*sqrt(2)], [2.0,-2.0,2.0], [2*sqrt(2),-2.0,0.0], [2.0,-2.0,-2.0], [0.0,-2.0,-2*sqrt(2)], [-2*sqrt(2),-2.0,0.0]
    ],

    'up': [ # up
        [0.0,-2*sqrt(2),2.0], [2.0,-2.0,2.0], [2*sqrt(2),0,2.0], [2.0,2.0,2.0], [0.0,2*sqrt(2),2.0], [-2*sqrt(2),0.0,2.0]
    ],
    
    'down': [ # down
        [0.0,-2*sqrt(2),-2.0], [2.0,-2.0,-2.0], [2*sqrt(2),0,-2.0], [2.0,2.0,-2.0], [0.0,2*sqrt(2),-2.0], [-2*sqrt(2),0.0,-2.0]
    ]
}


VIEW = {
    'front': [0, 1, 2, 3],
    'back': [4, 5, 6, 7],
    'right': [2, 3, 6, 7],
    'left': [0, 1, 4, 5],
    'up': [0, 3, 4, 7],
    'down': [1, 2, 5, 6]
}


def get_pkl(pkl_path, key):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data[key]


def look_at_rotation(camera_pos, target, up=np.array([0, 0, 1])):
    """
    计算从 camera_pos 看向 target 的旋转矩阵
    """
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)
    true_up = true_up / np.linalg.norm(true_up)

    # Blender camera looks along -Z, so invert forward
    rotation = np.stack([-right, true_up, -forward], axis=1)
    return rotation

        
def get_args():
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--shape_type', type=str, help='渲染对象类型')
    parser.add_argument('--ply_name', type=str, help='被渲染的ply文件名') 
    parser.add_argument('--view_path', type=str, help='视角pkl文件') 
    parser.add_argument('--center_path', type=str, help='中心pkl文件')
    parser.add_argument('--ply_dir', type=str, help='点云文件')
    parser.add_argument('--output_dir', type=str, help='保存路径')
    parser.add_argument('--correct_index', type=str, help='正确的索引')

    args = parser.parse_args()

    # args.view_path = '/home/lkh/siga/CADIMG/datasets/render_normal/views_correct.pkl'  # 视角pkl文件
    # args.center_path = '/home/lkh/siga/CADIMG/datasets/render_normal/centers_correct.pkl'  # 中心pkl文件
    # args.ply_dir = '/home/lkh/siga/dataset/deepcad/data/cad_ply/body2'
    # args.output_dir = '/home/lkh/siga/dataset/deepcad/data/cad_img/normal_body2'  # 渲染结果保存路径

    return args


def main():
    args = get_args()
    with open(args.correct_index, 'r') as f:
        data = json.load(f)
    for item in data:
        if item.get('explaination') == 'filter edge face':
            matched_index = item['file_names']
    if args.ply_name not in matched_index:
        print(f'------------------{args.ply_name}')
        return False

    bproc.init()
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # 读取要渲染的对象
    ply_path = os.path.join(args.ply_dir, args.shape_type, args.ply_name+'.ply')    
    objs = bproc.loader.load_obj(ply_path)
    obj=objs[0]
    obj.set_location([0,0,0])

    # 读取视角和中心
    view = get_pkl(args.view_path, args.ply_name)
    center = np.array(get_pkl(args.center_path, args.ply_name))
    camera_po = VIEW_CORNERS_6[view]

    # 添加相机
    for v in camera_po:
        camera_pos =  np.array(v)
        rotation = look_at_rotation(camera_pos, center)
        cam_pose = bproc.math.build_transformation_mat(camera_pos, rotation)
        bproc.camera.add_camera_pose(cam_pose)

    bproc.renderer.enable_normals_output()
    data = bproc.renderer.render()

    # 保存文件
    output_dir = os.path.join(args.output_dir, args.shape_type, args.ply_name)
    bproc.writer.write_hdf5(output_dir, data)



if __name__ == "__main__":
    main()
    
