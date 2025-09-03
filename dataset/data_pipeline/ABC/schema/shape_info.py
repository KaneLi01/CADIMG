import numpy as np
import trimesh
import pickle
import copy, os, sys, json
from math import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import utils.cadlib.Brep_utils as Brep_utils
from utils.vis import render_cad
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List


def compute_minmax_center(data: list[list[float]]) -> list[float]:
    '''用于计算多个child_num的全部最小bbox'''
    first_parts = [sub[:3] for sub in data]  # [[x, y, z], ...]
    second_parts = [sub[3:] for sub in data] # [[x, y, z], ...]

    # 计算每个位置上的最小值和最大值
    min_xyz = [min(coords) for coords in zip(*first_parts)]
    max_xyz = [max(coords) for coords in zip(*second_parts)]

    center = [(mi + ma) / 2 for mi, ma in zip(min_xyz, max_xyz)]

    return min_xyz + max_xyz, center


@dataclass
class CADFeature_Child:
    face_num: Optional[int] = None    
    wire_num: Optional[int] = None
    bbox_min_max: Optional[List[float]] = None
    bbox_center: Optional[List[float]] = None


@dataclass
class CADFeature:
    name: str
    valid: bool = False                  # step文件存在，且能正确读取
    child_num: Optional[int] = None 
    face_num: Optional[List[int]] = None    
    wire_num: Optional[List[int]] = None
    bbox_min_max: Optional[List[List[float]]] = None
    bbox_center: Optional[List[List[float]]] = None
    sub_shapes: List[CADFeature_Child] = field(default_factory=list, init=False)
    shape_volume: Optional[List[float]] = None  #  所有子形状的体积
    bbox_volume: Optional[List[float]] = None  # 包围盒的体积
    solid_valid: Optional[List[int]] = None  # 是否是solid
    faces_area: Optional[List[float]] = None  # 所有面的面积
    faces_type: Optional[List[int]] = None  # 所有面的类型
    edges_len: Optional[List[float]] = None  # 所有面的面积
    edges_type: Optional[List[int]] = None  # 所有面的类型    

    @classmethod
    def from_dict(cls, data: dict) -> "CADFeature":
        """从字典中创建"""
        return cls(
            name=data.get("name", ""),
            valid=data.get("valid", False),
            child_num=data.get("child_num"),
            face_num=data.get("face_num"),
            wire_num=data.get("wire_num"),
            bbox_min_max=data.get("bbox_min_max"),
            bbox_center=data.get("bbox_center"),
            shape_volume=data.get("shape_volume"),
            bbox_volume=data.get("bbox_volume"),
            solid_valid=data.get("solid_valid"),
            faces_area=data.get("faces_area"),
            faces_type=data.get("faces_type"),
            edges_len=data.get("edges_len"),
            edges_type=data.get("edges_type")        
        )

    def append_to_jsonl(self, jsonl_path: str):
        """将当前 CADFeature 实例追加写入到 jsonl 文件"""
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    def build_sub_shapes(self):
        """构建 sub_shapes 列表，每个子结构为 CADFeature_Child 实例"""
        if self.child_num is None:
            raise ValueError("child_num 不能为 None")
        if not all([self.face_num, self.wire_num, self.bbox_min_max, self.bbox_center]):
            raise ValueError("face_num, wire_num, bbox_min_max, bbox_center 不能为 None 或空")
        if not (len(self.face_num) == len(self.wire_num) == len(self.bbox_min_max) == len(self.bbox_center) == self.child_num):
            raise ValueError("各字段的长度应等于 child_num")

        self.sub_shapes.clear()
        for i in range(self.child_num):
            shape = CADFeature_Child(
                face_num=self.face_num[i],
                wire_num=self.wire_num[i],
                bbox_min_max=self.bbox_min_max[i],
                bbox_center=self.bbox_center[i]
            )
            self.sub_shapes.append(shape)

    def merge_sub_shapes(self):
        """child_num > 2，则将前 child_num-1 个子shape合并为一个"""
        # self.build_sub_shapes()
        if self.child_num is None or self.face_num is None or self.wire_num is None:
            raise ValueError("child_num / face_num / wire_num 不可为 None")
        
        if self.child_num <= 2:
            return self.face_num, self.wire_num

        # 合并前 child_num-1 个
        face_sum = sum(self.face_num[:-1])
        wire_sum = sum(self.wire_num[:-1])

        new_face_num = [face_sum, self.face_num[-1]]
        new_wire_num = [wire_sum, self.wire_num[-1]]

        new_minmax, new_center = compute_minmax_center(self.bbox_min_max[:-1])
        new_bbox_minmax = [new_minmax, self.bbox_min_max[-1]]
        new_bbox_center = [new_center, self.bbox_center[-1]]

        return CADFeature(
            name=self.name,
            valid=self.valid,
            child_num=2,
            face_num=new_face_num,
            wire_num=new_wire_num,
            bbox_min_max=new_bbox_minmax,     
            bbox_center=new_bbox_center        
        )

@dataclass
class CADFeature_Supp:
    name: str   
    min_dis: Optional[float] = None
    common_volume: Optional[float] = None


class Get_Feat_From_Dict():    
    '''从字典中读取CAD特征并筛选'''
    def __init__(self, feat):
        self.feat = CADFeature.from_dict(feat)
        self.valid = self.feat.valid
        self.name = self.feat.name
        self.child_num = self.feat.child_num
        self.face_num = self.feat.face_num
        self.wire_num = self.feat.wire_num
        self.bbox_min_max = self.feat.bbox_min_max
        self.bbox_center = self.feat.bbox_center
        self.shape_volume = self.feat.shape_volume
        self.bbox_volume = self.feat.bbox_volume
        self.solid_valid = self.feat.solid_valid
        self.faces_area = self.feat.faces_area
        self.faces_type = self.feat.faces_type
        self.edges_len = self.feat.edges_len
        self.edges_type = self.feat.edges_type

        if 'min_dis' in feat:
            self.min_dis = feat["min_dis"]
        if 'common_volume' in feat:
            self.common_volume = feat["common_volume"]
        if 'view' in feat:
            self.view = feat['view']

        self.sub_shapes = []

        if self.child_num:
            for i in range(self.child_num):
                shape = CADFeature_Child(
                    face_num=self.face_num[i],
                    wire_num=self.wire_num[i],
                    bbox_min_max=self.bbox_min_max[i],
                    bbox_center=self.bbox_center[i]
                )
                self.sub_shapes.append(shape)


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


class StepProcessor:
    def __init__(self, step_root_dir, output_root_dir):
        self.step_root_dir = step_root_dir
        self.output_root_dir = output_root_dir
    
    def get_shape_outputpath(self, name):
        """
        获取name的step文件的shape
        """
        # 构建输入输出路径
        step_cls = name[2:4]
        step_dir = os.path.join(self.step_root_dir, step_cls, name)
        step_file_name = os.listdir(step_dir)[0]
        step_path = os.path.join(step_dir, step_file_name)

        output_dir = os.path.join(self.output_root_dir, step_cls, name)
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, step_file_name.split('.')[0])  # 这里需要给输出路径添加后缀

        # 处理形状
        self.shape = Brep_utils.get_BRep_from_step(step_path)

        return self.shape, self.output_path

    def get_op_shape(self, face_num, child_num, dot_type):

        sub_shapes = Brep_utils.get_child_shapes(self.shape)
        
        # 根据条件选择操作形状
        if child_num == 2:
            # dot_type：是否翻转op shape
            choose_first = (face_num[0] <= face_num[1]) ^ (dot_type == -1)
            shape_op = sub_shapes[0] if choose_first else sub_shapes[1]
            
        elif 3 <= child_num <= 30:
            choose_combined = (sum(face_num[:-1]) <= face_num[-1]) ^ (dot_type == -1)
            shape_op = Brep_utils.make_compound(sub_shapes[:-1]) if choose_combined else sub_shapes[-1]
        else:
            raise ValueError(f"Unsupported child_num: {child_num}")
        
        return shape_op
    


def get_args():
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--name', type=str, help='名称')
    parser.add_argument('--num', type=int, help='渲染序号')


    args = parser.parse_args()


    return args


def test():
    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
    output_root_dir = '/home/lkh/siga/dataset/ABC/temp/obj'
    feat = {"name": "00000000", "min_dis": 0.0, "common_volume": 0.0, "valid": True, "child_num": 6, "face_num": [3, 3, 3, 11, 3, 2], "wire_num": [12, 18, 18, 48, 6, 7], "bbox_min_max": [[-58.271575682928116, -12.710093782594612, 246.30321323450605, -32.80108836795897, 12.710093782594608, 288.8785188706269], [-12.133108766389494, -49.145020786246874, 185.63703262411056, 79.24205348520054, 45.657349871394175, 391.2525921653276], [-82.1413255373635, -37.69221602081097, 332.5796598569866, -30.14216216139344, 24.31856247139173, 461.1744059983905], [-104.28464267599361, -0.6772699029366993, 103.88695409870931, 104.28464267599361, 104.30426980935307, 482.4097260830447], [-101.82686377134574, -101.81457659857693, -0.22686377134574356, 101.77771805221172, 101.81457659857693, 104.79108777299174], [-75.25010421212912, -75.2501041846798, 481.6332340954282, 75.2501042178524, 75.2501041846798, 536.3280950504528]], "bbox_center": [[-45.53633202544354, -1.7763568394002505e-15, 267.59086605256647], [33.55447235940552, -1.74383545742635, 288.44481239471907], [-56.14174384937847, -6.686826774709619, 396.87703292768856], [0.0, 51.813499953208186, 293.148340090877], [-0.02457285956700872, 0.0, 52.282112000823], [2.8616398140002275e-09, 0.0, 508.9806645729405]], "view": "up"}
    sp = StepProcessor(step_root_dir=step_root_dir, output_root_dir=output_root_dir)
    _, _ = sp.get_shape_outputpath(name='00000000')
    s = sp.get_op_shape(feat['face_num'], feat['child_num'], dot_type=-1)
    render_cad.display_BRep(s)


if __name__ == '__main__':
    test()