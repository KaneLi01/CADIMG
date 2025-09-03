import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import combinations


import utils.cadlib.Brep_utils as Brep_utils
import utils.jsonl_utils as jsonl_utils
from utils.vis import render_cad
from shape_info import Shapeinfo
from utils.cadlib.curves import *
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
from utils.cadlib.Brep_utils import Point
from dataclasses import dataclass, asdict
from typing import Optional, List
import argparse
import multiprocessing
from multiprocessing import TimeoutError


@dataclass
class CADFeature:
    name: str
    valid: bool = False                  # step文件存在，且能正确读取
    child_num: Optional[int] = None 
    face_num: Optional[List[int]] = None    
    wire_num: Optional[List[int]] = None
    bbox_min_max: Optional[List[List[float]]] = None
    bbox_center: Optional[List[List[float]]] = None


@dataclass
class SoilidFeature:
    face_num: Optional[int] = None    
    wire_num: Optional[int] = None
    bbox_min_max: Optional[List[float]] = None
    bbox_center: Optional[List[float]] = None


def append_feature_to_jsonl(feature, jsonl_path):
    """将 CADFeature 实例追加写入到 jsonl 文件"""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(asdict(feature)) + "\n")


def cad_info2json(args):
    """
    将CAD文件信息写入json文件
    """
    shape_info = Shape_Info_From_Step(args.step_path).shape_info

    append_feature_to_jsonl(shape_info, args.json_path)


def process_step(step_path):
    shape = Shape_Info_From_Step_2(step_path)
    return {
        'name': shape.cad_name,
        'min_dis': shape.min_dis,
        'common_volume': shape.common_volume
    }


class Shape_Info_From_Step():
    '''从step文件中读取Brep的基础信息'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]
        try:
            self.shape = copy.deepcopy(Brep_utils.get_BRep_from_step(self.step_path))
            self.child_num = self.shape.NbChildren()
            if self.child_num == 1:
                wire_num = [len(Brep_utils.get_wires(self.shape))]
                face_num = [len(Brep_utils.get_faces(self.shape))]
                bbox = Brep_utils.get_bbox(self.shape)
                min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z]]
                center = [[bbox.center.x, bbox.center.y, bbox.center.z]]
            elif self.child_num > 1:
                sub_shapes = Brep_utils.get_child_shapes(self.shape)
                wire_num = [len(Brep_utils.get_wires(sub_shape)) for sub_shape in sub_shapes]
                face_num = [len(Brep_utils.get_faces(sub_shape)) for sub_shape in sub_shapes]
                bboxs = [Brep_utils.get_bbox(sub_shape) for sub_shape in sub_shapes]
                min_max_pt = [[bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z] for bbox in bboxs]
                center = [[bbox.center.x, bbox.center.y, bbox.center.z] for bbox in bboxs]

            self.shape_info = CADFeature(
                name=self.cad_name,
                valid=True,
                child_num=self.child_num,
                face_num=face_num,
                wire_num=wire_num,
                bbox_min_max=min_max_pt,
                bbox_center=center,
            )
        except Exception as e:
            self.shape_info = CADFeature(name=self.cad_name)
            print(f"Error reading STEP file {self.step_path}: {e}")


class Shape_Info_From_Step_2():
    '''从step文件中读取Brep的补充信息'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]
        try:
            self.shape = copy.deepcopy(Brep_utils.get_BRep_from_step(self.step_path))
            self.sub_shapes = Brep_utils.get_child_shapes(self.shape)
            if len(self.sub_shapes) == 2:
                self.min_dis = round(Brep_utils.get_min_distance(self.sub_shapes[0], self.sub_shapes[1]), 8)

                common = BRepAlgoAPI_Common(self.sub_shapes[0], self.sub_shapes[1])
                common_shape = common.Shape()
                self.common_volume = Brep_utils.get_volume(common_shape)
            elif 2 < len(self.sub_shapes) <= 20:
                self.shape_base = Brep_utils.combine_shapes(self.sub_shapes[:-1]) 
                self.shape_op = self.sub_shapes[-1]

                self.min_dis = round(Brep_utils.get_min_distance(self.shape_base, self.shape_op), 8)

                common = BRepAlgoAPI_Common(self.shape_base, self.shape_op)
                common_shape = common.Shape()
                self.common_volume = Brep_utils.get_volume(common_shape)                

        except Exception as e:
            self.shape_info = CADFeature(name=self.cad_name)
            print(f"Error reading STEP file {self.step_path}: {e}")
        

class Filter_Name_From_feat_child():    
    '''从jsonl文件中读取CAD特征并筛选'''
    def __init__(self, feat):
        self.feat = feat
        self.child_num = self.feat["child_num"]
        self.face_num = self.feat["face_num"]
        self.wire_num = self.feat["wire_num"]
        self.bbox_min_max = self.feat["bbox_min_max"]
        self.bbox_center = self.feat["bbox_center"]

        self.sub_shapes = []

        for i in range(self.child_num):
            shape = SoilidFeature(
                face_num=self.face_num[i],
                wire_num=self.wire_num[i],
                bbox_min_max=self.bbox_min_max[i],
                bbox_center=self.bbox_center[i]
            )
            self.sub_shapes.append(shape)


    def filter_complex_faces(self, thre=20):
        '''筛选所有shape面数是否小于阈值'''
        for shape in self.sub_shapes:
            num_face = shape.face_num
            if num_face >= thre:
                return False
        return True

    def filter_complex_wires(self, thre=64):
        '''筛选所有shape线数是否小于阈值'''
        for shape in self.sub_shapes:
            num_wire = shape.wire_num
            if num_wire >= thre:
                return False
        return True
    
    def filter_small_thin(self, scale=10):
        '''筛选所有shape是否不会过小、是否不是薄面或棍'''
        def check_slice(dx, dy, dz, scale):
            '''如果任意两边长度比大于scale，则true'''
            for a, b in combinations([dx, dy, dz], 2):
                longer = max(a, b)
                shorter = min(a, b)
                if longer >= scale * shorter:
                    return True
            return False
        
        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]     
            dx, dy, dz = bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]

            if check_slice(dx, dy, dz, scale):
                return False
        return True


    # todo

    def get_view(self):
        '''利用包围盒计算观测角度，以让add的sketch不会被mask'''
        base_min, base_max = self.min_points[1], self.max_points[1]
        add_min, add_max = self.min_points[2], self.max_points[2]

        base_min = Point(0,0,0)
        base_max = Point(1,1,1)
        add_min = Point(0,0,1)
        add_max = Point(1,1,2)

        if base_max.z - add_min.z <= 1e-5:
            view = 'up' 
        elif base_min.z - add_max.z >= -1e-5:
            view = 'down'
        elif base_max.x - add_min.x <= 1e-5:
            view = 'front'
        elif base_min.x - add_max.x >= -1e-5:
            view = 'back'
        elif base_max.y - add_min.y <= 1e-5:
            view = 'right'
        elif base_min.y - add_max.y >= -1e-5:
            view = 'left'
        else: raise Exception('please check code')

        return view
        

    def filter_intersection(self, thre=1e-5):
        '''筛选两个body是否不相交'''
        common = BRepAlgoAPI_Common(self.shapes[1], self.shapes[2])
        common_shape = common.Shape()
        common_volume = Brep_utils.get_volume(common_shape)
        if common_volume < thre:
            return True
        else: return False

    def filter_distance(self, thre=1e-5):
        '''筛选两个body最小距离是否极小'''
        min_dis = Brep_utils.get_min_distance(self.shapes[1], self.shapes[2])
        if min_dis < thre:
            return True
        else: return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, default=None, help="Path to the STEP file")
    parser.add_argument("--json_path", type=str, default=None, help="Path to the output json file")
    parser.add_argument("--cls_idx", type=str, default='00', help="Class index for filtering")
    parser.add_argument("--child_num", type=str, default='2', help="Class index for filtering")

    args = parser.parse_args()
    return args


def cad_info2json2(cls_idx, child_num):

    step_dir = '/home/lkh/siga/dataset/ABC/temp/step'

    if child_num == '2':        
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child2_simple_boxy.jsonl'
        output_jsonl = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child2_simple_boxy_supp.jsonl'
    if child_num == '20':
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child3_20_simple_boxy.jsonl'
        output_jsonl = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child3_20_simple_boxy_supp.jsonl'        

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


def filter_jsonl_from_child():
    child_txt = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child_num_3up.txt'
    child_feats_jsonl = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child3_20.jsonl'
    names = []
    with open(child_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去掉换行符和首尾空格
            names.append(line)

    for name in names:        
        feat = jsonl_utils.read_jsonl_from_name(name)
        jsonl_utils.append_dic(child_feats_jsonl, dict=feat.feats)


def filter1():
    '''从jsonl文件中筛选出符合条件的CAD特征，并写入新的jsonl文件'''
    child_num = 20
    if child_num == 2:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child2.jsonl'
        out_jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child2_simple_boxy.jsonl'
    
    elif child_num == 20:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child3_20.jsonl'
        out_jsonl_path = '/home/lkh/siga/CADIMG/dataset/ABC_feat/child3_20_simple_boxy.jsonl'        

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                feat = Filter_Name_From_feat_child(obj)
                if feat.filter_complex_faces(thre=30) and feat.filter_complex_wires(thre=128) and feat.filter_small_thin(scale=15):
                    jsonl_utils.append_dic(out_jsonl_path, dict=obj)
            except (json.JSONDecodeError, ValueError) as e:
                print(e)
                return 0
    

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
    args = get_args()
    cad_info2json2(cls_idx=args.cls_idx, child_num=args.child_num)


if __name__ == "__main__":
    main()