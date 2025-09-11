import sys, os, copy, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import random
import numpy as np
from math import pi
import ast
from scipy.stats import truncnorm

import utils.cadlib.Brep_utils as Brep_utils
from utils.path_file_utils import read_txt_lines
import utils.vis.render_cad as render_cad
import utils.path_file_utils as path_file_utils
from dataset.prepare_data.ABC.shape_info import VIEW_CORNERS_6

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Dir, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend.DataExchange import write_step_file

class Childnum1BaseAddMerge():
    def __init__(self, output_root_dir=None, base_txt_path=None, add_txt_paths=None, process_method='render', output_view_path=None, weights=None):
        
        self.base_txt_path = base_txt_path
        if base_txt_path is not None:
            self.base_names_list = read_txt_lines(self.base_txt_path)
        self.add_txt_paths = add_txt_paths
        self.output_root_dir = output_root_dir
        self.output_view_path = output_view_path  # 视角输出路径
        if weights == None and add_txt_paths is not None:
            self.weights = [1] * len(add_txt_paths)
        else:
            self.weights = weights

        # 注册处理方法
        self.process_registry = {}
        self._register_default_methods()

        self.current_process_method = process_method

    def _register_default_methods(self):
        """注册默认的处理方法"""
        self.register_process_method('render', self._process_render)
        self.register_process_method('save', self._process_save_step)
        self.register_process_method('None', self._process_none)

    def register_process_method(self, name, method):
        """
        注册处理方法
        
        Args:
            name (str): 方法名称
            method (callable): 处理方法，接受参数 (name, base_shape_nor, target_shape, view)
        """
        self.process_registry[name] = method

    def process_all_shapes(self, interrupt=None):
        # 处理并保存base/add shape, 的mesh和step，保存到路径下的step/mesh，二级目录base/add，三级直接记录文件

        os.makedirs(self.output_root_dir, exist_ok=True)
        if self.output_view_path is not None:
            if os.path.exists(self.output_view_path):
                os.remove(self.output_view_path)

        # 遍历所有shape
        if self.current_process_method not in self.process_registry:
            raise ValueError(f"Process method '{self.current_process_method}' not registered")
        
        process_func = self.process_registry[self.current_process_method]  

        for i, name in enumerate(self.base_names_list):
            if interrupt is not None:
                if name <= interrupt:
                    continue
            print(f"Processing shape: {name}")
            try:
                base_shape = self.read_shape_from_name(name)

                add_shape, add_name = self.get_random_add_shape()

                # 对两个shape进行归一化
                base_shape_nor = Brep_utils.normalize_shape(base_shape)
                add_shape_nor = Brep_utils.normalize_shape(add_shape)
                
                # 获得刚性变换后的add和对应视角
                # 调整大小和偏移？那有洞的怎么办？
                add_shape_srt, view = self.compute_trans_add_shape_view(base_shape_nor, add_shape_nor)

                process_func(name, base_shape_nor, add_shape_srt, view, add_name)
            except Exception as e:
                continue

    def _process_render(self, name, base_shape, add_shape, view, add_name=None):
        """将合并好的图像渲染，用于临时查看"""
        campos = VIEW_CORNERS_6[view]
        output_path = os.path.join(self.output_root_dir, f'{name}_{add_name}.png')
        render_cad.save_BRep_list(output_path, shape=[base_shape, add_shape],
                                cam_pos=campos[1], see_at=[0,0,0], bg_color=1.0)
    
    def _process_save_step(self, name, base_shape, add_shape, view, add_name=None):
        # 直接在这一步保存obj时有问题的，在保存step文件后另再保存obj
        path_file_utils.append_line_to_file(self.output_view_path, f'({name}, {view})')

        directories = [
            os.path.join(self.output_root_dir, 'step', 'base'),
            os.path.join(self.output_root_dir, 'step', 'add')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        output_path_step_base = os.path.join(directories[0], f'{name}.step')
        output_path_step_add = os.path.join(directories[1], f'{name}.step')
        Brep_utils.save_Brep_to_step(base_shape, output_path_step_base)
        Brep_utils.save_Brep_to_step(add_shape, output_path_step_add)

    def _process_none(self, name, base_shape, add_shape, view, add_name=None):
        pass

    def get_random_add_shape(self):
        txt_path = random.choices(self.add_txt_paths, weights=self.weights, k=1)[0]
        add_names_list = read_txt_lines(txt_path)
        if not add_names_list:
            raise Exception(f"No names found in {txt_path}")
        add_name = random.choice(add_names_list)
        add_shape = self.read_shape_from_name(add_name)

        return add_shape, add_name

    def read_shape_from_name(self, name):
        step_rootdir = '/home/lkh/siga/dataset/ABC/step'
        name_class = name[2:4]
        step_dir = os.path.join(step_rootdir, name_class, name)
        step_path_rela = os.listdir(step_dir)[0]
        step_path = os.path.join(step_dir, step_path_rela)

        return Brep_utils.get_BRep_from_step(step_path)

    def compute_trans_add_shape_view(self, base_shape, add_shape, scale_factor=None, thre=0):
        '''通过base和add shape，将二者移动到相切位置，然后返回移动后的add shape和相机视角'''
        # 1. add shape 随机缩放

        base_pl, base_bpl = Brep_utils.get_valid_biggest_plane(base_shape)  # 根据面积确定缩放尺寸
        max_thre = Brep_utils.get_face_area(base_pl) / Brep_utils.get_face_area(base_bpl)
        
        trsf_s = gp_Trsf()
        if scale_factor is None:
            scale_factor = self.truncated_normal_sample(max_thre)
        trsf_s.SetScale(gp_Pnt(0, 0, 0), scale_factor)  
        add_shape_s = BRepBuilderAPI_Transform(add_shape, trsf_s, True).Shape()  # 缩放后的add shape

        
        _, add_s_bpl = Brep_utils.get_valid_biggest_plane(add_shape_s, thre=thre)

        base_bpl_center = self.get_face_center_vec(base_bpl)  # 面心向量和坐标轴平行（后续还需验证）
        add_s_bpl_center = self.get_face_center_vec(add_s_bpl)

        view = self.vec2view(base_bpl_center)  # 获取视角

        rot_axis = base_bpl_center.Crossed(add_s_bpl_center)  # 旋转轴
        add_bpl_sr = add_s_bpl
        add_shape_sr = add_shape_s
        if (rot_axis.X(), rot_axis.Y(), rot_axis.Z()) != (0.0, 0.0, 0.0):
            # base 和 add 不共线的情况， 旋转到共线
            trsf_r = gp_Trsf()
            origin = gp_Pnt(0, 0, 0)
            rot_dir = gp_Dir(rot_axis.X(), rot_axis.Y(), rot_axis.Z())
            rot_axis = gp_Ax1(origin, rot_dir)
            trsf_r.SetRotation(rot_axis, pi/2)
            add_bpl_sr = BRepBuilderAPI_Transform(add_s_bpl, trsf_r).Shape()
            add_shape_sr = BRepBuilderAPI_Transform(add_shape_s, trsf_r).Shape()
        
        add_bpl_sr_center = self.get_face_center_vec(add_bpl_sr)

        # add shape平移
        trsf_t = gp_Trsf()
        dot_product = add_bpl_sr_center.Dot(base_bpl_center)
        if dot_product == 0:
            raise ValueError("Vectors are orthogonal, cannot determine translation direction")

        if dot_product < 0:
            sign = -1
        else: 
            sign = 1
            trsf_f = gp_Trsf()
            trsf_f.SetMirror(gp_Pnt(0.0, 0.0, 0.0))
            add_shape_sr = BRepBuilderAPI_Transform(add_shape_sr, trsf_f).Shape()
        t = gp_Vec(
            base_bpl_center.X() + sign * add_bpl_sr_center.X(),
            base_bpl_center.Y() + sign * add_bpl_sr_center.Y(),
            base_bpl_center.Z() + sign * add_bpl_sr_center.Z()
        )  
        trsf_t.SetTranslation(t)   
        add_shape_srt = BRepBuilderAPI_Transform(add_shape_sr, trsf_t, True).Shape()

        return add_shape_srt, view
        
    def get_face_center_vec(self, face):
        """
        输入：一个 TopoDS_Face （正方形）
        输出：中心点 gp_Pnt
        """
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        center = props.CentreOfMass()  # 返回 gp_Pnt

        x = center.X()
        y = center.Y()
        z = center.Z()
        max_val = max((x, y, z), key=abs)
        x_new, y_new, z_new = [
            val if abs(val) == abs(max_val) else 0
            for val in (x, y, z)
        ]
        
        return gp_Vec(x_new, y_new, z_new)

    def truncated_normal_sample(self, max_thre):
        """
        生成满足正态分布N的随机样本
        """
        # 定义截断范围
        low = 0.3
        mean = 0.5
        std = 0.8
        if max_thre < low:
            high = low + 0.01
        else:
            high = max_thre
        
        # 计算标准化后的截断边界
        a = (low - mean) / std
        b = (high - mean) / std
        
        # 生成截断正态分布样本
        samples = truncnorm.rvs(a, b, loc=mean, scale=std)
        
        return samples
    
    def vec2view(self, vec):
        if vec.X() > 0.0:
            return 'front'
        elif vec.X() < 0.0:
            return 'back'
        elif vec.Y() > 0.0:
            return 'right'
        elif vec.Y() < 0.0:
            return 'left'
        elif vec.Z() > 0.0:
            return 'up'
        elif vec.Z() < 0.0:
            return 'down'
        else:
            raise ValueError("wrong vec")

def process_shape_old():
    mode = 'render'  # 'render', 'save_mesh', 'save_step'

    idx_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/random_index.txt'
    output_view_path = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/view.txt'
    os.makedirs('/home/lkh/siga/dataset/ABC/step_imgs/childnum1_base_add_shape/1', exist_ok=True)
    base_txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/base_valid_plane/base_valid_plane.txt'
    add_txt_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/add_complexity'
    add_txt_class = ['low', 'medium', 'high', 'super']
    add_txt_paths = []
    for atc in add_txt_class:
        add_txt_path = os.path.join(add_txt_dir, atc + '.txt')
        if not os.path.exists(add_txt_path):
            raise Exception(f"File {add_txt_path} does not exist.")
        add_txt_paths.append(add_txt_path)

    if mode == 'render':
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_imgs/merge_shapes1/1/temp'
        path_file_utils.recreate_dir(output_root_dir)        
        rb = Childnum1BaseAddMerge(output_root_dir=output_root_dir, 
                                base_txt_path=base_txt_path, 
                                add_txt_paths=add_txt_paths, 
                                process_method='render')
    elif mode == 'save':
        output_root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1'
        path_file_utils.recreate_dir(output_root_dir)  
        rb = Childnum1BaseAddMerge(output_root_dir=output_root_dir, 
                                base_txt_path=base_txt_path, 
                                add_txt_paths=add_txt_paths, 
                                process_method='save',
                                output_view_path=output_view_path)       
    
    rb.process_all_shapes()


def process_shape(args):
    mode = args.mode

    base_txt_path = '/home/lkh/siga/dataset/ABC/shape_feats/processed/new/v3/filter_base_valid_plane/1/valid.txt'
    add_txt_dir = '/home/lkh/siga/dataset/ABC/shape_feats/processed/new/v3/distinguish_add_complexity/1'
    # 各自的数量：    272     229      1111      6611 5278 3643 9768     3663      4748 6359 6512 ; 12other:645
    add_txt_class = ['cone', 'torus', 'sphere', '3', '4', '5', '6cube', '6other', '7', '8', '9up']
    weights       = [0.4   , 0.3    , 0.5     , 1.1, 1.1, 1.1, 0.8    , 1       , 0.9, 0.8, 0.8  ]
    add_txt_paths = []
    for atc in add_txt_class:
        add_txt_path = os.path.join(add_txt_dir, atc + '.txt')
        if not os.path.exists(add_txt_path):
            raise Exception(f"File {add_txt_path} does not exist.")
        add_txt_paths.append(add_txt_path)

    if mode == 'render':
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_imgs/merge_shapes1/1/temp'
        output_view_path = os.path.join(output_root_dir, 'view.txt')
        path_file_utils.recreate_dir(output_root_dir)      
        rb = Childnum1BaseAddMerge(output_root_dir=output_root_dir, 
                                base_txt_path=base_txt_path, 
                                add_txt_paths=add_txt_paths, 
                                process_method='render',
                                weights=weights)
    elif mode == 'save':
        output_root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1'
        output_view_path = os.path.join(output_root_dir, 'view.txt')
        if args.interrupt is None:
            path_file_utils.recreate_dir(output_root_dir)  
        rb = Childnum1BaseAddMerge(output_root_dir=output_root_dir, 
                                base_txt_path=base_txt_path, 
                                add_txt_paths=add_txt_paths, 
                                process_method='save',
                                output_view_path=output_view_path,
                                weights=weights)       
    
    rb.process_all_shapes(interrupt=args.interrupt)


def test_merge_method():
    step_path1 = '/home/lkh/siga/dataset/ABC/step/72/00721530/00721530_e1580f24a1edf4c3a28ccb3f_step_000.step'
    step_path2 = '/home/lkh/siga/dataset/ABC/step/16/00163517/00163517_23e4ee0b9011c50dc889ecc6_step_001.step'
    s1 = Brep_utils.normalize_shape(Brep_utils.get_BRep_from_step(step_path1))
    s2 = Brep_utils.normalize_shape(Brep_utils.get_BRep_from_step(step_path2))

    # render_cad.display_BRep_list_with_different_color([s1, s2], ['blue', 'green'])

    cp = Childnum1BaseAddMerge()
    add_shape_srt, view = cp.compute_trans_add_shape_view(s1, s2)
    # render_cad.display_BRep_list_with_different_color([s1, add_shape_srt], ['blue', 'green'])
    Brep_utils.save_Brep_to_step(s1, '/home/lkh/siga/CADIMG/dataset/process/ABC/temp.step')


def main(args):
    process_shape(args)
    # test_merge_method()
    # output_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/random_index.txt'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="render|save")
    parser.add_argument("--interrupt", type=str, default=None)  # 中断数
    args = parser.parse_args()
    main(args)