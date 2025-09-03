import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from itertools import combinations
from dataclasses import dataclass, asdict
from typing import Optional, List
import argparse
import multiprocessing
from multiprocessing import TimeoutError
import shutil

from dataset.prepare_data.ABC.shape_info import CADFeature, CADFeature_Child, CADFeature_Supp, Get_Feat_From_Dict
from dataset.data_pipeline.ABC.prepare.extract_feature_json import Shape_Info_From_Step
from utils.vis.merge_imgs import select_merge_imgs
import utils.cadlib.Brep_utils as Brep_utils
import utils.jsonl_utils as jsonl_utils
from utils.vis import render_cad
from utils.cadlib.curves import *
from utils.jsonl_utils import JsonlOperator, BaseJsonlHandler
from utils.path_file_utils import write_list_to_txt, recreate_dir
from utils.cadlib.Brep_utils import Point
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape




class Filter_Name_From_Feat(Get_Feat_From_Dict):    
    '''从字典中读取CAD特征并筛选'''
    def filter_null(self):
        if not self.valid:
            return False
        else: return True

    def filter_child_num(self):
        if self.child_num > 20:
            return False
        else: return True

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

    def filter_simple_faces(self, thre=10):
        '''筛选至少一个面数小于阈值'''
        if self.child_num == 2:
            for shape in self.sub_shapes:
                num_face = shape.face_num
                if num_face <= thre:
                    return True
            return False 
        if 3 <= self.child_num <=20:
            n1 = 0
            for shape in self.sub_shapes[:-1]:
                num_face = shape.face_num
                n1 += num_face
            n2 = self.sub_shapes[-1].face_num
            if n1 <= thre or n2 <= thre:
                return True
            else: return False
    
    def filter_small_thin(self, scale=10):
        '''筛选所有shape是否不是薄面或棍'''
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
    
    def filter_stick(self, scale=10):
        '''筛选所有 shape 是否不是棍状'''
        def is_stick(dx, dy, dz, scale):
            """最长边如果是其他两个边的 scale 倍及以上，就判定为棍"""
            edges = [dx, dy, dz]
            longest = max(edges)
            edges.remove(longest)  # 剩下两个短边
            return all(longest >= scale * e for e in edges)

        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]
            dx, dy, dz = (
                bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2],
            )

            if is_stick(dx, dy, dz, scale):
                return False  # 棍状 → False
        return True  # 全部不是棍状 → True    

    def filter_zero_face(self):
        '''筛选掉child不是shape的情况'''
        if self.child_num == 2:
            if self.sub_shapes[0].face_num == 0 or self.sub_shapes[1].face_num == 0:
                return False
            else: return True    
            
        if 3 <= self.child_num <=20:
            f1 = 0
            for sub_shape in self.sub_shapes:
                f = sub_shape.face_num
                f1 += f
            f2 = self.sub_shapes[-1].face_num

            if f1 == 0 or f2 == 0:
                return False 
            else: return True 

    def filter_intersection(self, thre=1):
        '''筛选两个body是否不相交'''
        if self.common_volume >= thre or self.common_volume <= -thre:
            return False
        else: return True

    def filter_distance(self, thre=1):
        '''筛选两个body最小距离是否极小'''
        if self.min_dis >= thre or self.min_dis <= -thre:
            return False
        else: return True
    
    def filter_simple_face_wire(self):
        '''筛选掉face为0, 线条数过少的情况'''
        for shape in self.sub_shapes:
            if (shape.face_num == 0) or (shape.wire_num < 4):
                return False
        return True
    
    def filter_shape_bbox_volume_ratio(self, ratio=4):
        '''筛选掉所有子shape，和其包围盒体积比值过小的情况。可以改善如桶状的对象'''
        for sv, bv in zip(self.shape_volume, self.bbox_volume):
            if sv*ratio < bv:
                return False
        return True
    
    def filter_solid(self):
        '''筛选所有子形状都为solid'''
        for sva in self.solid_valid:
            if sva == 0:
                return False
        return True
    
    def filter_screw(self, thre=15, cond=False):
        '''筛选螺丝类型的形状'''  
        if (any(x <= 2 for x in self.face_num)):
            return True  
        if (any(2 < x <= 4 for x in self.face_num) and ((3 in self.faces_type) or (4 in self.faces_type))):
            return True
        for edges_len, face_type in zip(self.edges_len, self.faces_type):
            if cond:
                if face_type == 1 or face_type == 6 or face_type == 7:
                    longest = max(edges_len)
                    shortest = min(edges_len)
                    if shortest*thre < longest:
                        return False
            else:
                longest = max(edges_len)
                shortest = min(edges_len)
                if shortest*thre < longest:
                    return False                
        return True
    
    def filter_edge_len(self, thre=1/8):
        '''归一化后最短的edge长度'''
        transposed = list(zip(*self.bbox_min_max)) 
        result = []
        for i in range(3):
            result.append(min(transposed[i]))
        # 后三个位置取最大值
        for i in range(3, 6):
            result.append(max(transposed[i]))

        dx, dy, dz = result[3] - result[0], result[4] - result[1], result[5] - result[2]
        longest_bbox_edge = max(dx, dy, dz)

        scale = longest_bbox_edge / 1.5

        for edges_len in self.edges_len:
            for edge_len in edges_len:
                if edge_len / (scale + 0.000000001) < thre:
                    return False
        return True

    def filter_face_area(self, thre=1/4):
        '''归一化后有plane，且至少有一个plane的面积大于thre'''
        transposed = list(zip(*self.bbox_min_max)) 
        result = []
        for i in range(3):
            result.append(min(transposed[i]))
        for i in range(3, 6):
            result.append(max(transposed[i]))

        dx, dy, dz = result[3] - result[0], result[4] - result[1], result[5] - result[2]
        bbox_face_area = [dx*dy, dx*dz, dy*dz]
        longest_bbox_face_area = max(bbox_face_area)

        for face_area, face_type in zip(self.faces_area, self.faces_type):
            if face_type == 0:
                if face_area / longest_bbox_face_area > thre:
                    return True
        return False

    def classify_plane(self, thre=10):
        min_bbox_face = []
        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]     
            dx, dy, dz = bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]     
            min_bbox_face.append(min(dx*dy, dx*dz, dy*dz))  
        minimal_bbox_face = min(min_bbox_face)

        for area, type in zip(self.faces_area, self.faces_type):
            if type == 0:
                if area*thre > minimal_bbox_face:
                    return True
                
        return False


class RawJsonlFilter():
    '''通过最原始的jsonl文件，进行过滤'''
    def __init__(self, root_dir: str, filter_func, removed_name_output_txt=None, filtered_output_jsonl=None):
        self.root_dir = root_dir
        self.filter_func = filter_func
        self.removed_name_output_txt = removed_name_output_txt
        self.filtered_output_jsonl = filtered_output_jsonl

    def filter_all(self) -> None:
        """
        按照顺序处理jsonl文件
        """
        match_list = []
        unmatch_list_name = []

        for i in range(100):
            filename = f"{i:02d}.jsonl"
            filepath = os.path.join(self.root_dir, filename)
            result = self.filter_one_file(filepath)
            if self.removed_name_output_txt:
                unmatch_list_name.extend(result["unmatched_name"])
            if self.filtered_output_jsonl:
                match_list.extend(result["matched"])
        
        if self.removed_name_output_txt:
            delete_list = []
            for item in unmatch_list_name:
                delete_list.append(item)
            write_list_to_txt(delete_list, self.removed_name_output_txt)
        
        if self.filtered_output_jsonl:
            jsonl_utils.write_dict_list(self.filtered_output_jsonl, match_list)

    def filter_one_file(self, filepath: str):
        self.jop = JsonlOperator(filepath)
        self.handler = self.jop.handler     

        result = self.jop.filter_by_func(self.filter_func)
        return result

    def get_valid_files(self):
        """
        检查jsonl文件是否存在
        """
        valid_files = []
        for i in range(100):
            filename = f"{i:02d}.jsonl"
            filepath = os.path.join(self.root_dir, filename)
            if os.path.isfile(filepath):
                valid_files.append(filename)
        
        return valid_files


class JsonlFilter():
    '''通过child分类后的jsonl文件，进行过滤'''
    def __init__(self, jsonl_path: str, filter_func, filtered_name_output_txt=None, filtered_output_jsonl=None, removed_name_output_txt=None, removed_output_jsonl=None):
        self.jsonl_path = jsonl_path
        self.filter_func = filter_func
        self.filtered_name_output_txt = filtered_name_output_txt
        self.filtered_output_jsonl = filtered_output_jsonl
        self.removed_name_output_txt = removed_name_output_txt
        self.removed_output_jsonl = removed_output_jsonl

        self.jop = JsonlOperator(self.jsonl_path)
        self.handler = self.jop.handler     

    def filter(self) -> None:
        match_list_name = []
        match_list = []
        unmatch_list_name = []
        unmatch_list = []
        
        result = self.jop.filter_by_func(self.filter_func)

        if self.filtered_name_output_txt:
            match_list_name.extend(result["matched_name"])
            write_list_to_txt(match_list_name, self.filtered_name_output_txt)
        if self.filtered_output_jsonl:
            match_list.extend(result["matched"])
            jsonl_utils.write_dict_list(self.filtered_output_jsonl, match_list)
        if self.removed_name_output_txt:
            unmatch_list_name.extend(result["unmatched_name"])
            write_list_to_txt(unmatch_list_name, self.removed_name_output_txt)
        if self.removed_output_jsonl:
            unmatch_list.extend(result["unmatched"])
            jsonl_utils.write_dict_list(self.removed_output_jsonl, unmatch_list)


class JsonlClassify():
    '''通过child多分类后的jsonl文件，进行过滤'''
    def __init__(self, jsonl_path: str, filter_func, category_paths=None):
        self.jsonl_path = jsonl_path
        self.filter_func = filter_func
        self.category_paths = category_paths
        self.jop = JsonlOperator(self.jsonl_path)
        self.handler = self.jop.handler     

    def filter(self) -> None:
        
        result = self.jop.classify_by_func(self.filter_func)
        for i, (category, items) in enumerate(result.items()):
            cat_paths = self.category_paths[category]
            names = [d for d in items['names']]
            feats = [f for f in items['items']]

            # 如果需要按类别保存到 txt
            output_txt = getattr(self, f"{category}_name_output_txt", cat_paths['txt'])
            if output_txt:
                write_list_to_txt(names, output_txt)

            # 如果需要按类别保存到 jsonl
            output_jsonl = getattr(self, f"{category}_output_jsonl", cat_paths['jsonl'])
            if output_jsonl:
                jsonl_utils.write_dict_list(output_jsonl, feats)


def filter_init_feat(): 
    '''从初始的100个jsonl文件中根据child num过滤'''
    mode = 'filter_child_num_3'  # del_null_child_num | filter_child_num_1/2/3 | 
    raw_jsonl_root_dir = '/home/lkh/siga/dataset/ABC/all/raw'
    
    removed_name_output_txt = None
    filtered_output_jsonl = None
    if mode == 'del_null_child_num':
        removed_name_output_txt = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/null.txt'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            return (feat.filter_null() and feat.filter_child_num())
    elif mode == 'filter_child_num_1':
        filtered_output_jsonl = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1.jsonl'
        def filter_func(data):
            return data.get("child_num") == 1   
    elif mode == 'filter_child_num_2':
        filtered_output_jsonl = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_2.jsonl'
        def filter_func(data):
            return data.get("child_num") == 2         
    elif mode == 'filter_child_num_3':
        filtered_output_jsonl = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_3.jsonl'
        def filter_func(data):
            child_num = data.get("child_num", -1)  # 如果缺失，默认为 -1
            if child_num == None:
                return False
            else: return 3 <= child_num <= 20    

    rjf = RawJsonlFilter(
        raw_jsonl_root_dir, 
        filter_func=filter_func,
        removed_name_output_txt=removed_name_output_txt, 
        filtered_output_jsonl=filtered_output_jsonl
        )
    rjf.filter_all()    


def filter_sub_feat(): 
    '''从记录更多特征的jsonl文件中过滤'''
    mode = 'screw'  # filter_face_wire_thin | filter_simple_face_wire | filter_shape_bbox_volume_ratio | filter_volume_thin2 | solid_valid | face_thin
    
    removed_name_output_txt = None
    filtered_output_jsonl = None
    filtered_name_output_txt = None

    for i in range(1, 4):
        if mode == 'filter_face_wire_thin':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return (feat.filter_complex_faces(thre=30) and feat.filter_complex_wires(thre=128) and feat.filter_small_thin(scale=15))   
        elif mode == 'filter_simple_face_wire':
            '''筛选掉face为0, 线条数过少的情况'''
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin_simple.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin_simple.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_facewirethin_simple_name.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_simple_face_wire() 
        elif mode == 'filter_shape_bbox_volume_ratio':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_volume/child_num_{i}_volume.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin_simple_volume.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin_simple_volume.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_facewirethin_simple_volume_name.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_shape_bbox_volume_ratio(ratio=4)
        elif mode == 'filter_volume_thin2':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin_simple_volume_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin2_simple_volume_all.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_facewirethin2_simple_volume_name.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_small_thin(scale=8)
        elif mode == 'solid_valid':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_solid/child_num_{i}_solid.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_solid/child_num_{i}_solidvalid.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume_repeat_solidvalid.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_facewirethin2_simple_volume_repeat_solidvalid.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_solid()
        elif mode == 'face_thin':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_{i}_face_edge_pro_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin2_simple_volume_all_repeat_facethin.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume_all_repeat_facethin.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_facewirethin2_simple_volume_all_repeat_facethin.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_screw(thre=50, cond=True)            
        elif mode == 'thin_screw':
            '''只有thin'''
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_{i}_face_edge_pro_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_thin_screw2.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_thin_screw2.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_thin_screw2.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                # return (feat.filter_screw(thre=50) and feat.filter_small_thin(scale=8))
                return feat.filter_small_thin(scale=5)
        elif mode == 'stick':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_{i}_face_edge_pro_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_stick.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_stick.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_stick.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_stick(scale=3)
        elif mode == 'screw':
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_{i}_face_edge_pro_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_screw.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_screw.txt'
            filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_{i}_screw.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                return feat.filter_screw(thre=50)           
                
        jf = JsonlFilter(
            jsonl_path=jsonl_path, 
            filter_func=filter_func,
            removed_name_output_txt=removed_name_output_txt, 
            filtered_output_jsonl=filtered_output_jsonl,
            filtered_name_output_txt=filtered_name_output_txt
            )
        jf.filter()        


def filter_same_step():
    '''如果有相同几何信息的shape，将其过滤'''
    select = 2
    if not hasattr(filter_same_step, "feat_set"):
        filter_same_step.feat_set = set()

    removed_name_output_txt = None
    filtered_output_jsonl = None
    filtered_name_output_txt = None

    for i in range(1, 4):
        if select == 1:
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin2_simple_volume_all.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_facewirethin2_simple_volume_all_repeat.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume_all_repeat.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                print(feat.name)
                feat_dic = {
                    'face_num': feat.face_num,
                    'wire_num': feat.wire_num,
                    'bbox_min_max': feat.bbox_min_max,
                    'bbox_center': feat.bbox_center,
                    'sub_shapes': feat.sub_shapes,
                    'shape_volume': feat.shape_volume,
                    'bbox_volume': feat.bbox_volume               
                }
                if feat_dic in filter_same_step.feat_list:
                    return False
                else:
                    filter_same_step.feat_list.append(feat_dic)
                    return True        
        elif select == 2:
            if i > 1:
                break
            jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}.jsonl'
            filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_{i}_repeat.jsonl'
            removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_repeat.txt'
            def filter_func(data):
                feat = Filter_Name_From_Feat(data)
                print(feat.name)
                feat_tuple = (
                    tuple(feat.face_num),
                    tuple(feat.wire_num),
                    tuple(feat.bbox_min_max[0]),
                    tuple(feat.bbox_center[0])
                )
                feat_hash = hash(feat_tuple)

                if feat_hash in filter_same_step.feat_set:
                    return False
                else:
                    filter_same_step.feat_set.add(feat_hash)
                    return True      
                    
        jf = JsonlFilter(
            jsonl_path=jsonl_path, 
            filter_func=filter_func,
            removed_name_output_txt=removed_name_output_txt, 
            filtered_output_jsonl=filtered_output_jsonl,
            filtered_name_output_txt=filtered_name_output_txt
            )
        jf.filter()        


def classify_face_type():
    '''根据face的类型分类，用于执行不同的add操作,只需要对child num为1的执行即可'''
    filtered_output_jsonl = None


    jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1.jsonl'
    filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1_facewirethin.jsonl'
    def classify_func(data):
        '''对shape是否有足够大的平面用于add进行分类'''
        feat = Filter_Name_From_Feat(data)
        return (feat.filter_complex_faces(thre=30) and feat.filter_complex_wires(thre=128) and feat.filter_small_thin(scale=15))        

    jf = JsonlFilter(
        jsonl_path=jsonl_path, 
        filter_func=classify_func,
        filtered_output_jsonl=filtered_output_jsonl,
        )
    jf.filter()        


def filter_final_feat_test():
    '''根据条件，筛选最后的结果，不做删除'''
    cond = 2  # 
    filtered_output_jsonl = None
    filtered_name_output_txt = None

    if cond == 1:
        jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.jsonl'
        filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/feat/child_num_1_1.jsonl'
        filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_1.txt'

        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            return (
                feat.filter_complex_faces(thre=20) 
                and feat.filter_complex_wires(thre=64) 
                and feat.filter_small_thin(scale=5) 
                and feat.filter_screw(thre=30) 
                and feat.filter_edge_len(thre=1/8)
                )   
    elif cond == 2:
        jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.jsonl'
        filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/feat/child_num_1_2.jsonl'
        filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_2.txt'
        removed_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/other/child_num_1_2.txt'


        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            return (
                feat.filter_complex_faces(thre=20)  # 小
                and feat.filter_complex_wires(thre=64)  # 小
                and feat.filter_small_thin(scale=5)  # 小
                and feat.filter_screw(thre=20)  # 小
                and feat.filter_edge_len(thre=1/6)  # 大
                and feat.filter_shape_bbox_volume_ratio(ratio=4)  # 小
                )  
    elif cond == 3:
        jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.jsonl'
        filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/feat/child_num_1_2.jsonl'
        filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_2.txt'

        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            return (
                feat.filter_complex_faces(thre=15)  # 小
                and feat.filter_complex_wires(thre=32)  # 小
                and feat.filter_small_thin(scale=4)  # 小
                and feat.filter_screw(thre=10)  # 小
                and feat.filter_edge_len(thre=1/6)  # 大
                and feat.filter_shape_bbox_volume_ratio(ratio=3)  # 小
                )  
    elif cond == 4:
        jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1_final1.jsonl'
        filtered_output_jsonl = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/feat/child_num_1_2.jsonl'
        filtered_name_output_txt = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_2.txt'

        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            return (
                feat.filter_complex_faces(thre=15)  # 小
                and feat.filter_complex_wires(thre=32)  # 小
                and feat.filter_small_thin(scale=4)  # 小
                and feat.filter_screw(thre=10)  # 小
                and feat.filter_edge_len(thre=1/6)  # 大
                and feat.filter_shape_bbox_volume_ratio(ratio=3)  # 小
                )      


    jf = JsonlFilter(
        jsonl_path=jsonl_path, 
        filter_func=filter_func,
        filtered_output_jsonl=filtered_output_jsonl,
        filtered_name_output_txt=filtered_name_output_txt,
        removed_name_output_txt=removed_name_output_txt
        )
    jf.filter()  


def classify_final_feat(
        cond=1,  # 一级路径命名
        jsonl_path='/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1_final1.jsonl',
        filter_func=None,  # 传入分类函数
        classify_dir_path=None,  # 二级路径命名
        classify_name=None,  # 传入分类的元组，第一个是保留的名称，第二个是过滤掉的名称
        output_root='/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1',
        ):        
    # name和feat保存路径
    if classify_dir_path is not None:
        sub_root_dir = os.path.join(str(cond), classify_dir_path)
    else:
        sub_root_dir = str(cond)
    output_root_dir = os.path.join(output_root, sub_root_dir)
    recreate_dir(output_root_dir)

    filtered_name = classify_name[0]
    removed_name = classify_name[1]

    filtered_output_jsonl = os.path.join(output_root_dir, f'{filtered_name}.jsonl')
    filtered_name_output_txt = os.path.join(output_root_dir, f'{filtered_name}.txt')
    removed_output_jsonl = os.path.join(output_root_dir, f'{removed_name}.jsonl')
    removed_name_output_txt = os.path.join(output_root_dir, f'{removed_name}.txt')

    jf = JsonlFilter(
        jsonl_path=jsonl_path, 
        filter_func=filter_func,
        filtered_output_jsonl=filtered_output_jsonl,
        filtered_name_output_txt=filtered_name_output_txt,
        removed_output_jsonl=removed_output_jsonl,
        removed_name_output_txt=removed_name_output_txt
        )
    jf.filter()  
    print('filter finished')

    # 图像保存路径
    render_root_dir = '/home/lkh/siga/dataset/ABC/step_imgs/child_num_1_final1'
    filtered_save_dir1 = os.path.join(render_root_dir, sub_root_dir, f'{filtered_name}0')
    filtered_save_dir2 = os.path.join(render_root_dir, sub_root_dir, f'{filtered_name}')
    removed_save_dir1 = os.path.join(render_root_dir, sub_root_dir, f'{removed_name}0')
    removed_save_dir2 = os.path.join(render_root_dir, sub_root_dir, f'{removed_name}')
    recreate_dir(filtered_save_dir1)
    recreate_dir(filtered_save_dir2)
    recreate_dir(removed_save_dir1)
    recreate_dir(removed_save_dir2)
    
    # 绘制add
    select_merge_imgs(
        save_dir1 = filtered_save_dir1,
        save_dir2 = filtered_save_dir2,
        txt_file = filtered_name_output_txt,
        n = 1620
    )

    # 绘制base
    select_merge_imgs(
        save_dir1 = removed_save_dir1,
        save_dir2 = removed_save_dir2,
        txt_file = removed_name_output_txt,
        n = 1620
    )    

    shutil.rmtree(filtered_save_dir1)
    shutil.rmtree(removed_save_dir1)


def classify_final_feat_multi(
        cond=1,
        jsonl_path='/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1_final1.jsonl',
        filter_func=None,
        classify_dir_path=None,
        classify_name=None,  # 多类别列表
        render=False
    ):
    # 基础输出路径
    output_root = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1'
    sub_root_dir = os.path.join(str(cond), classify_dir_path) if classify_dir_path else str(cond)
    output_root_dir = os.path.join(output_root, sub_root_dir)
    recreate_dir(output_root_dir)

    # 渲染保存目录
    render_root_dir = '/home/lkh/siga/dataset/ABC/step_imgs/child_num_1_final1'

    # 创建类别到路径的映射
    category_paths = {}
    for category in classify_name:
        category_paths[category] = {
            "jsonl": os.path.join(output_root_dir, f"{category}.jsonl"),
            "txt": os.path.join(output_root_dir, f"{category}.txt"),
            "img_tmp": os.path.join(render_root_dir, sub_root_dir, f"{category}_temp"),
            "img_final": os.path.join(render_root_dir, sub_root_dir, category)
        }
        if render:
            recreate_dir(category_paths[category]["img_tmp"])
            recreate_dir(category_paths[category]["img_final"])
    

    jc = JsonlClassify(
        jsonl_path=jsonl_path, 
        filter_func=filter_func,
        category_paths=category_paths
        )
    jc.filter()
    print('filter finished')

    # 绘制图片
    if render:
        for category in classify_name:
            select_merge_imgs(
                save_dir1=category_paths[category]["img_tmp"],
                save_dir2=category_paths[category]["img_final"],
                txt_file=category_paths[category]["txt"],
                n=1620
            )
            shutil.rmtree(category_paths[category]["img_tmp"])

    # print("多分类完成")



def classify_base_add():
    '''根据条件，筛选base shape 和 add shape。child num 为1的情况。筛选后同时绘图'''
    cond = 2
    classify_name = ('add', 'base')
    if cond == 1:
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)

            if (feat.filter_complex_faces(thre=10) 
                and feat.filter_complex_wires(thre=32) 
                and feat.filter_small_thin(scale=3) 
                and feat.filter_screw(thre=20) 
                and feat.filter_edge_len(thre=1/5)
                ):
                return True
            elif feat.filter_complex_faces(thre=4):
                return True
            else:
                return False 
        classify_final_feat(cond=cond, filter_func=filter_func, classify_name=classify_name) 
    elif cond == 2:
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)

            if (feat.filter_complex_faces(thre=16) 
                and feat.filter_complex_wires(thre=32) 
                and feat.filter_small_thin(scale=3) 
                and feat.filter_screw(thre=20) 
                and feat.filter_edge_len(thre=1/5)
                ):
                return True
            elif feat.filter_complex_faces(thre=4):
                return True
            else:
                return False 
        classify_final_feat(cond=cond, filter_func=filter_func, classify_name=classify_name, output_root='/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/0824')        


def classify_base_feat():
    '''进一步筛选处有效的base，根据plane'''
    def get_step_path_from_name(name):
        step_dir = '/home/lkh/siga/dataset/ABC/step'
        cls_idx = name[2:4]

        step_subdir = os.path.join(step_dir, cls_idx, name)
        step_file = os.listdir(step_subdir)
        step_path = os.path.join(step_subdir, step_file[0])
        return step_path

    sub_cond = 'base_big_plane'
    if sub_cond == 'base_valid_plane':
        classify_name = ('base_valid_plane', 'base_unvalid_plane')
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/base.jsonl'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            print(f'processing {feat.name}')
            step_path = get_step_path_from_name(feat.name)
            sifs = Shape_Info_From_Step(step_path)
            if feat.filter_face_area(thre=1/4) and (sifs.get_valid_biggest_plane() is not None):
                return True
            else:
                return False     
    elif sub_cond == 'base_big_plane':
        '''和上面进行对比，不适用'''
        classify_name = ('base_big_plane', 'base_small_plane')
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/base.jsonl'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            print(f'processing {feat.name}')
            if feat.filter_face_area(thre=1/4):
                return True
            else:
                return False    
    
    classify_final_feat(
        cond=1,  # 一级路径命名
        jsonl_path=jsonl_path,
        filter_func=filter_func,  # 传入分类函数
        classify_dir_path=sub_cond,  # 二级路径命名
        classify_name=classify_name  # 传入分类的元组，第一个是保留的名称，第二个是过滤掉的名称
        )


def classify_add_feat():
    '''筛选add'''
    sub_cond = 'add_conplexity'
    render = True      
    if sub_cond == 'add_face_num_all':
        classify_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        render = False
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/add.jsonl'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            for i in range(1, 10):
                if feat.face_num[0] == i:
                    return str(i)
    elif sub_cond == 'add_face_num_all/6':
        classify_name = ['cube', 'other']
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/add_face_num_all/6.jsonl'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            if abs(1 - feat.bbox_volume[0] / feat.shape_volume[0]) < 0.001:
                return 'cube'
            else:
                return 'other'
    elif sub_cond == 'add_conplexity':
        # low: 123 ; medium: 45; high: 6cube; super: 6other789
        classify_name = ['low', 'medium', 'high', 'super']
        render = True
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/add.jsonl'
        def filter_func(data):
            feat = Filter_Name_From_Feat(data)
            if feat.face_num[0] <= 3:
                return 'low'
            elif 3 < feat.face_num[0] <= 5:
                return 'medium'
            elif feat.face_num[0] == 6 and abs(1 - feat.bbox_volume[0] / feat.shape_volume[0]) < 0.001:
                return 'high'
            elif (feat.face_num[0] == 6 and abs(1 - feat.bbox_volume[0] / feat.shape_volume[0]) >= 0.001) or feat.face_num[0] >= 7:
                return 'super'

    classify_final_feat_multi(
        cond=1,  # 一级路径命名
        jsonl_path=jsonl_path,
        filter_func=filter_func,  # 传入分类函数
        classify_dir_path=sub_cond,  # 二级路径命名
        classify_name=classify_name,  # 传入分类的列表
        render=render
        )


def main():
    
    # filter_sub_feat()
    #filter_final_feat_test()
    # classify_add_feat()
    #classify_base_add()
    filter_same_step()

if __name__ == "__main__":
    main()