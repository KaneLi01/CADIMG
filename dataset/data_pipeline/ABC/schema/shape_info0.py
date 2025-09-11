import numpy as np
import trimesh
import pickle
import copy, os, sys
from math import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils.cadlib.Brep_utils as Brep_utils
from utils.vis import render_cad
import argparse


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
    
        seq = Brep_utils.get_seq_from_json(shape_path)
        seq1 = copy.deepcopy(seq)
        self.shape = Brep_utils.get_BRep_from_seq(seq1.seq[-1:])
        # self.wires = Brep_utils.get_wireframe(self.shape)
        self.wires = Brep_utils.get_wireframe_cir(self.shape)


def get_args():
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--name', type=str, help='名称')
    parser.add_argument('--num', type=int, help='渲染序号')


    args = parser.parse_args()


    return args