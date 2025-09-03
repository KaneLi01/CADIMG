
import blenderproc as bproc
import numpy as np
import trimesh
import pickle
import os
import argparse
from math import sqrt
import json
import sys
import copy

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


def get_pkl(path, name):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict[name]


class Shapeinfo:
    def __init__(self, shape_name, view_num):
        self.shape_name = shape_name
        self.view_num = view_num

        json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
        shape_path = os.path.join(json_dir, self.shape_name[:4], self.shape_name+'.json')

        pkl_dir = '/home/lkh/siga/CADIMG/dataset/create_dataset/render_normal'
        center_path = os.path.join(pkl_dir, 'centers_correct.pkl')
        view_path = os.path.join(pkl_dir, 'views_correct.pkl')

        self.seeat = get_pkl(center_path, shape_name)
        view = get_pkl(view_path, shape_name)
        self.campos = VIEW_CORNERS_6[view][view_num]




def get_args():
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--name', type=str, help='名称')
    parser.add_argument('--num', type=int, help='渲染序号')


    args = parser.parse_args()


    return args


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

        

def blender(name, center, campos):
    bproc.init()
    bproc.world.set_world_background_hdr_img('/home/lkh/siga/CADIMG/experiments/test/white.hdr')

    sketch_ply_dir = '/home/lkh/siga/dataset/my_dataset/cad_ply/addbody/operate'
    ply_path = os.path.join(sketch_ply_dir, name+'.ply')
    objs = bproc.loader.load_obj(ply_path)
    obj=objs[0]
    obj.set_location([0,0,0])

    campos = np.array(campos)
    rotation = look_at_rotation(campos, np.array(center))
    cam_pose = bproc.math.build_transformation_mat(campos, rotation)
    bproc.camera.add_camera_pose(cam_pose)

    bproc.renderer.enable_normals_output()
    data = bproc.renderer.render()

    output_dir = '/home/lkh/siga/CADIMG/experiments/test/output'
    output_path = os.path.join(output_dir, name)
    bproc.writer.write_hdf5(output_path, data)

def main():
    args = get_args()
    shape = Shapeinfo(args.name, args.num)
    blender(args.name, shape.seeat, shape.campos)


if __name__ == "__main__":
    main()
    
