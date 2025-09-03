import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from itertools import combinations


import utils.cadlib.Brep_utils as Brep_utils
import utils.jsonl_utils as jsonl_utils
from utils.rootdir_processer import FileProcessor
from utils.path_file_utils import write_list_to_txt
from utils.vis import render_cad
from dataset.prepare_data.ABC.shape_info import CADFeature, CADFeature_Child, CADFeature_Supp, Get_Feat_From_Dict
from utils.cadlib.curves import *
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
from utils.cadlib.Brep_utils import Point
from dataclasses import dataclass, asdict
import argparse
import multiprocessing
from multiprocessing import TimeoutError
from typing import List, Dict, Tuple, Optional
import traceback


def append_feature_to_jsonl(feature, jsonl_path):
    """将 CADFeature 实例追加写入到 jsonl 文件"""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(asdict(feature)) + "\n")


def process_step(step_path):
    shape = Shape_Info_From_Step_2(step_path)
    return {
        'name': shape.cad_name,
        'min_dis': shape.min_dis,
        'common_volume': shape.common_volume
    }


class StepDir2Jsonl(FileProcessor):
    '''把step文件的不同类型的信息写入jsonl文件'''
    def __init__(self, root_dir: str, feat_type='init'):
        super().__init__(root_dir, extension=".step")
        self.feat_type = feat_type  # init | tang | init_child_num | 'volume'
        
    def process_file(self, input_filepath: str, output_path=None) -> None:
        shape_info = Shape_Info_From_Step(str(input_filepath))
        
        if self.feat_type == 'init':
            '''按照类，如00，11，22划分'''
            name_class = shape_info.cad_name[2:4]
            root_dir = '/home/lkh/siga/dataset/ABC/all/raw_1'
            os.makedirs(root_dir, exist_ok=True)

            output_path = root_dir + '/' + f'{name_class}.jsonl'
            cad_feat = shape_info.get_init_feat()
        elif self.feat_type == 'init_child_num':
            '''按照child_num划分'''
            child_num = shape_info.child_num
            root_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/init_feat_child_num'
            os.makedirs(root_dir, exist_ok=True)
            if child_num == 1 or child_num == 2:
                output_path = root_dir + '/' + f'child_num_{child_num}_init.jsonl'
            elif 2 < child_num < 20:
                output_path = root_dir + '/' + 'child_num_3_init.jsonl'
            cad_feat = shape_info.get_init_feat()
        elif self.feat_type == 'volume':
            '''按照child_num划分'''
            child_num = shape_info.child_num
            root_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_volume'
            os.makedirs(root_dir, exist_ok=True)
            if child_num == 1 or child_num == 2:
                output_path = root_dir + '/' + f'child_num_{child_num}_volume.jsonl'
            elif 2 < child_num:
                output_path = root_dir + '/' + 'child_num_3_volume.jsonl'
            cad_feat = shape_info.get_volume_feat()   
        elif self.feat_type == 'solid':
            '''是否只有face'''         
            child_num = shape_info.child_num
            root_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_solid'
            os.makedirs(root_dir, exist_ok=True)
            if child_num == 1 or child_num == 2:
                output_path = root_dir + '/' + f'child_num_{child_num}_solid.jsonl'
            elif 2 < child_num:
                output_path = root_dir + '/' + 'child_num_3_solid.jsonl'
            cad_feat = shape_info.get_vaild_solid()  
        elif self.feat_type == 'face_edge_pro':
            '''face和edge的更多信息，如面积，长度，类型'''
            child_num = shape_info.child_num
            root_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro'
            os.makedirs(root_dir, exist_ok=True)
            if child_num == 1 or child_num == 2:
                output_path = root_dir + '/' + f'child_num_{child_num}_face_edge_pro.jsonl'
            elif 2 < child_num:
                output_path = root_dir + '/' + 'child_num_3_solid_face_edge_pro.jsonl'
            cad_feat = shape_info.get_face_edge_info_pro()

        jsonl_utils.append_dic(output_path, dict=cad_feat)
        

class Shape_Info_From_Step():
    '''从step文件中读取Brep的基础信息'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]  # cad名称
        self.shape = copy.deepcopy(Brep_utils.get_BRep_from_step(self.step_path))  # 总形状
        self.sub_shapes = Brep_utils.get_child_shapes(self.shape)  # 子形状列表
        self.child_num = len(self.sub_shapes)  # 子形状数量
        
    def get_init_feat(self):
        try:
            wire_num = [len(Brep_utils.get_wires(sub_shape)) for sub_shape in self.sub_shapes]
            face_num = [len(Brep_utils.get_faces(sub_shape)) for sub_shape in self.sub_shapes]
            bboxs = [Brep_utils.get_bbox(sub_shape) for sub_shape in self.sub_shapes]
            min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z] for bbox in bboxs]
            center = [[bbox.center.x, bbox.center.y, bbox.center.z] for bbox in bboxs]
            shape_volume = [Brep_utils.get_volume(sub_shape) for sub_shape in self.sub_shapes]  # 所有子形状的体积
            bbox_volume = [(mm[3]-mm[0])*(mm[4]-mm[1])*(mm[5]-mm[2]) for mm in min_max_pt]  # 所有包围盒的体积

            cad_feat = {
                'name':self.cad_name,
                'valid':True,
                'child_num':self.child_num,
                'face_num':face_num,
                'wire_num':wire_num,
                'bbox_min_max':min_max_pt,
                'bbox_center':center,
                'shape_volume':shape_volume,
                'bbox_volume':bbox_volume
            }

        except Exception as e:
            cad_feat = {
                'name':self.cad_name,
                'valid':False,
                'child_num':None,
                'face_num':None,
                'wire_num':None,
                'bbox_min_max':None,
                'bbox_center':None,
                'shape_volume':None,
                'bbox_volume':None                
            }
            traceback.print_exc()
        
        return cad_feat

    def get_volume_feat(self):
        try:
            bboxs = [Brep_utils.get_bbox(sub_shape) for sub_shape in self.sub_shapes]
            min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z] for bbox in bboxs]
            shape_volume = [Brep_utils.get_volume(sub_shape) for sub_shape in self.sub_shapes]  # 所有子形状的体积
            bbox_volume = [(mm[3]-mm[0])*(mm[4]-mm[1])*(mm[5]-mm[2]) for mm in min_max_pt]  # 所有包围盒的体积

            cad_feat = {
                'name':self.cad_name,
                'shape_volume':shape_volume,
                'bbox_volume':bbox_volume
            }

        except Exception as e:
            cad_feat = {
                'name':self.cad_name,
                'shape_volume':None,
                'bbox_volume':None                
            }
            traceback.print_exc()
        
        return cad_feat
    
    def get_vaild_solid(self):
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
        try:
            solid_valid = []
            if self.child_num != 1:
                for sub_shape in self.sub_shapes:
                    exp_solid = TopExp_Explorer(sub_shape, TopAbs_SOLID)
                    if exp_solid.More():
                        solid_valid.append(1)
                    else:
                        solid_valid.append(0)
            else:
                exp_solid = TopExp_Explorer(self.shape, TopAbs_SOLID)
                if exp_solid.More():
                    solid_valid.append(1)
                else:
                    solid_valid.append(0)
            
            cad_feat = {
                'name':self.cad_name,
                'solid_valid':solid_valid,
            }


        except Exception as e:
            cad_feat = {
                'name':self.cad_name,
                'solid_valid':None,
            }
            traceback.print_exc()
        
        return cad_feat       

    def get_face_edge_info_pro(self):
        '''获取所有face的面积和类型，还有face中的所有edge，并记录edge的长度和类型'''
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        try:
            face_area_list, face_type_list = Brep_utils.get_faces_area_type(self.shape)
            face_list = Brep_utils.get_faces(self.shape)
            edge_len_list = []
            edge_type_list = []
            for face in face_list:
                edge_len_from_face = []
                edge_type_from_face = []
                edges = Brep_utils.get_edges_from_face(face)
                for edge in edges:
                    curve = BRepAdaptor_Curve(edge)
                    curve_type = curve.GetType()
                    edge_type_from_face.append(curve_type)  # 记录边的类型

                    props = GProp_GProps()
                    brepgprop.LinearProperties(edge, props)
                    length = props.Mass()
                    edge_len_from_face.append(length)  # 记录边的长度 
                edge_len_list.append(edge_len_from_face)  
                edge_type_list.append(edge_type_from_face)
            
            cad_feat = {
                'name':self.cad_name,
                'faces_area':face_area_list,
                'faces_type':face_type_list,
                'edges_len':edge_len_list,
                'edges_type':edge_type_list
            }


        except Exception as e:
            cad_feat = {
                'name':self.cad_name,
                'faces_area':None,
                'faces_type':None,
                'edges_len':None,
                'edges_type':None
            }
            traceback.print_exc()
        
        return cad_feat               

    def get_tangency_feat(self):
        try:
            if self.child_num == 2:
                self.min_dis = round(Brep_utils.get_min_distance(self.sub_shapes[0], self.sub_shapes[1]), 8)

                common = BRepAlgoAPI_Common(self.sub_shapes[0], self.sub_shapes[1])
                common_shape = common.Shape()
                self.common_volume = Brep_utils.get_volume(common_shape)
            elif 2 < self.child_num <= 20:
                self.shape_base = Brep_utils.combine_shapes(self.sub_shapes[:-1]) 
                self.shape_op = self.sub_shapes[-1]

                self.min_dis = round(Brep_utils.get_min_distance(self.shape_base, self.shape_op), 8)

                common = BRepAlgoAPI_Common(self.shape_base, self.shape_op)
                common_shape = common.Shape()
                self.common_volume = Brep_utils.get_volume(common_shape)                

        except Exception as e:
            self.shape_info = CADFeature(name=self.cad_name)
            print(f"Error reading STEP file {self.step_path}: {e}")

    def get_valid_biggest_plane(self, thre=1/4):
        if self.child_num != 1:
            return None
        
        shape_scaled = Brep_utils.normalize_shape(self.shape)
        face_area_list, face_type_list = Brep_utils.get_faces_area_type(shape_scaled)

        if 0 in face_type_list:
            face_list = Brep_utils.get_faces(shape_scaled)
            bbox = Brep_utils.get_bbox(shape_scaled)
            min_p = bbox.min
            max_p = bbox.max
            
            dx, dy, dz = max_p.x - min_p.x, max_p.y - min_p.y, max_p.z - min_p.z
            bbox_face_area = [dx*dy, dx*dz, dy*dz]
            longest_bbox_face_area = max(bbox_face_area)
            valid_faces = []
            valid_faces_area = []

            # 获取bbox的所有面对象
            bbox_shape = Brep_utils.create_box_from_minmax(min_p, max_p)
            bbox_faces = Brep_utils.get_faces(bbox_shape)
        else:
            return None

        for face, area, t in zip(face_list, face_area_list, face_type_list):
            if t == 0:
                if area / longest_bbox_face_area > thre:
                    valid_faces.append(face)
                    valid_faces_area.append(area)
        # 按照area大到小排列
        sorted_faces = [face for face_area, face in sorted(zip(valid_faces_area, valid_faces), key=lambda x: x[0], reverse=True)]
        
        for face in sorted_faces:
            for bface in bbox_faces:
                if Brep_utils.is_planes_equal(face, bface):
                    return (face, bface)
        
        return None
                


class Shape_Info_From_Step_1():
    '''从step文件中读取Brep的体积和包围盒体积'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]    


# class Shape_Info_From_Step_2():
#     '''从step文件中读取Brep的补充信息'''
#     def __init__(self, step_path):
#         self.step_path = step_path
#         self.cad_name = self.step_path.split('/')[-1].split('_')[0]
#         try:
#             self.shape = copy.deepcopy(Brep_utils.get_BRep_from_step(self.step_path))
#             self.sub_shapes = Brep_utils.get_child_shapes(self.shape)
#             if len(self.sub_shapes) == 2:
#                 self.min_dis = round(Brep_utils.get_min_distance(self.sub_shapes[0], self.sub_shapes[1]), 8)

#                 common = BRepAlgoAPI_Common(self.sub_shapes[0], self.sub_shapes[1])
#                 common_shape = common.Shape()
#                 self.common_volume = Brep_utils.get_volume(common_shape)
#             elif 2 < len(self.sub_shapes) <= 20:
#                 self.shape_base = Brep_utils.combine_shapes(self.sub_shapes[:-1]) 
#                 self.shape_op = self.sub_shapes[-1]

#                 self.min_dis = round(Brep_utils.get_min_distance(self.shape_base, self.shape_op), 8)

#                 common = BRepAlgoAPI_Common(self.shape_base, self.shape_op)
#                 common_shape = common.Shape()
#                 self.common_volume = Brep_utils.get_volume(common_shape)                

#         except Exception as e:
#             self.shape_info = CADFeature(name=self.cad_name)
#             print(f"Error reading STEP file {self.step_path}: {e}")
        


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, default=None, help="Path to the STEP file")
    parser.add_argument("--json_path", type=str, default=None, help="Path to the output json file")
    parser.add_argument("--cls_idx", type=str, default='00', help="Class index for filtering")
    parser.add_argument("--child_num", type=str, default='2', help="Class index for filtering")

    args = parser.parse_args()
    return args


def cad_info2json():
    '''从step root目录中读取所有shape的基本信息'''
    t = 5

    step_root_dir = '/home/lkh/siga/dataset/ABC/step'
    if t == 1:
        sp = StepDir2Jsonl(root_dir=step_root_dir, feat_type='init')  # 实例化step2jsonl类，输出目录在类中定义
    elif t == 2:
        sp = StepDir2Jsonl(root_dir=step_root_dir, feat_type='init_child_num')
    elif t == 3:
        sp = StepDir2Jsonl(root_dir=step_root_dir, feat_type='volume')
    elif t == 4:
        sp = StepDir2Jsonl(root_dir=step_root_dir, feat_type='solid')
    elif t == 5:
        sp = StepDir2Jsonl(root_dir=step_root_dir, feat_type='face_edge_pro')

    sp.process_all()  # conti参数，从哪个数字继续


def find_miss_name():
    '''从step文件直接读取shape时，会有各种意外情况使其无法写入jsonl文件中。获得这部分没有写入的name，写在txt中方便删去'''
    total_jsonl_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/'
    new_jsonl_dir = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_volume/'
    output_txt = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/reading_step_miss.txt'

    miss_names = []
    for i in range(1,4):
        total_jsonl_path = total_jsonl_dir + f'child_num_{i}_facewirethin_simple.jsonl'
        new_jsonl_path = new_jsonl_dir + f'child_num_{i}_volume.jsonl'
        tj_jo = jsonl_utils.JsonlOperator(total_jsonl_path)
        nj_jo = jsonl_utils.JsonlOperator(new_jsonl_path)

        total_names = tj_jo.load_name_to_list()
        new_names = nj_jo.load_name_to_list()

        miss_name = sorted(list(set(total_names) - set(new_names)))
        miss_names.extend(miss_name)
    write_list_to_txt(miss_names, output_txt)
        


def cad_info2json1():
    '''从step root目录中读取Brep的体积和包围盒体积'''
    step_root_dir = '/home/lkh/siga/dataset/ABC/step'
    output_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/test.jsonl'
    sp = StepDir2Jsonl(root_dir=step_root_dir)
    sp.process_all(output_path=output_path)

def cad_info2json2(cls_idx, child_num):

    step_dir = '/home/lkh/siga/dataset/ABC/temp/step'

    if child_num == '2':        
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/child2_simple_boxy.jsonl'
        output_jsonl = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/child2_simple_boxy_supp2.jsonl'
    if child_num == '20':
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/child3_20_simple_boxy.jsonl'
        output_jsonl = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/child3_20_simple_boxy_supp2.jsonl'        

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            obj = json.loads(line)
            if obj['name'][2:4] != cls_idx:
                continue

            name = obj['name']
            step_subdir = os.path.join(step_dir, cls_idx, name)
            step_files = os.listdir(step_subdir)
            step_path = os.path.join(step_subdir, step_files[0])
            print(f"Processing {step_path}")

            with multiprocessing.Pool(processes=1) as pool:
                try:
                    result = pool.apply_async(process_step, (step_path,))
                    feat = result.get(timeout=15)
                    jsonl_utils.append_dic(output_jsonl, dict=feat)
                except TimeoutError:
                    print(f"[超时跳过] 第 {line_num} 行 {name}")
                except Exception as e:
                    print(f"[错误跳过] 第 {line_num} 行 {name} 报错: {e}")




def write_jsonl():
    try:
        args = get_args()
        cad_info2json(args)

    except Exception as e:
        print(f"[Error] 程序执行出错: {e}", file=sys.stderr)
        sys.exit(1)  # 非0表示出错    


def merge_simpleboxyface_supp(child_num=2):
    '''将过滤后的json进行字典合并。合并补充信息，并删去多余键'''
    if child_num == 2:
        json1 = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child2_simple_boxy_supp.jsonl'
        json2 = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child2_simple_boxy_face.jsonl'
        output_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/child2_final0.jsonl'
    elif child_num == 20:
        json1 = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child3_20_simple_boxy_supp.jsonl'
        json2 = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child3_20_simple_boxy_face.jsonl'
        output_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/child3_20_final0.jsonl'
    
    jsonl_utils.merge_jsonls(json1, json2, output_path)


def test():
    # path = '/home/lkh/siga/dataset/ABC/temp/step/68/00680353/00680353_58e46a26ffd7440ff2757add_step_000.step'
    # path = '/home/lkh/siga/dataset/ABC/temp/step/68/00680066/00680066_974dfa07171a18719254fced_step_001.step'
    path = '/home/lkh/siga/dataset/ABC/temp/step/68/00680023/00680023_58e4403663d6ac0f6a35b509_step_001.step'
    s = Shape_Info_From_Step_2(path)


def main():
    # test()
    # filter1()
    # shape = Brep_utils.get_BRep_from_step('/home/lkh/siga/dataset/ABC/temp/step/68/00680000/00680000_58e44015b4d9da0f6f9fdbf2_step_009.step')
    # render_cad.display_BRep(shape)
    # args = get_args()
    # cad_info2json2(cls_idx=args.cls_idx, child_num=args.child_num)
    # merge_simpleboxyface_supp(child_num=20)
    cad_info2json()


if __name__ == "__main__":
    main()