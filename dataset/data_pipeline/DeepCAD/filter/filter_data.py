import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from itertools import combinations


import utils.cadlib.Brep_utils as Brep_utils
import utils.cadlib.Brep_utils_test as Brep_utils_t
from utils.vis import render_cad
from shape_info import Shapeinfo
from utils.cadlib.curves import *
from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.TopoDS import TopoDS_Shape
from utils.cadlib.Brep_utils import Point


class Shape_Info_From_Json():
    '''从Json文件中读取Brep的基础序列信息'''
    def __init__(self, cad_json_path):
        self.cad_json_path = cad_json_path
        self.cad_name = self.cad_json_path.split('/')[-1].split('.')[0]
        self.cad_seq = Brep_utils.get_seq_from_json(self.cad_json_path)
        self.cad_body_len = len(self.cad_seq.seq)
        if self.cad_body_len > 1:
            # 注意在使用seq创建shape时，要用复制的seq，否则尺度会出错
            self.sub_seqs = [self.cad_seq.seq, self.cad_seq.seq[:-1], self.cad_seq.seq[-1:]]            
        else: 
            self.sub_seqs = [self.cad_seq.seq]


class Filter_Name_From_Seq(Shape_Info_From_Json):
    '''从Brep序列中读取信息，进行筛选等处理'''
    def __init__(self, cad_json_path):
        super().__init__(cad_json_path)
        
        self.last_seq = self.sub_seqs[-1][0]
        self.profile = self.last_seq.profile
        self.loop = self.profile.children[0]
        self.curve = self.loop.children[0]

    def filter_2body(self):
        '''筛选是否是两个body合并的shape'''
        if self.cad_body_len == 2:
            return True
        else: return False

    def filter_union(self):
        '''筛选两个body是否是并集操作'''
        if self.last_seq.operation == 0 or self.last_seq.operation == 1:
            return True
        else: return False

    def filter_circle(self):
        '''筛选proflie中是否包含圆形'''
        if isinstance(self.curve, Circle):
            return True
        else: return False
        

class Filter_Name_From_Shape(Shape_Info_From_Json):
    '''从shape出发进行筛选等处理'''
    def __init__(self, cad_json_path):
        super().__init__(cad_json_path)

        self.shapes = [Brep_utils.get_BRep_from_seq(sub_seq) for sub_seq in self.sub_seqs]
        bboxs_temp = [Brep_utils.get_bbox(shape) for shape in self.shapes]
        self.centers = [bbox.center for bbox in bboxs_temp]
        self.min_points = [bbox.min for bbox in bboxs_temp]
        self.max_points = [bbox.max for bbox in bboxs_temp]
        self.bboxs = [Brep_utils.create_box_from_minmax(min_pt, max_pt) 
                      for min_pt, max_pt in zip(self.min_points, self.max_points)]
    
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
        

    def filter_complex_faces(self, thre=30):
        '''筛选shape面数是否小于阈值'''
        num_face = len(Brep_utils.get_faces(self.shapes[0]))
        if num_face < thre:
            return True
        else: return False

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
    
    def filter_small_thin(self, shortest=0.04, scale=15):
        '''筛选shape和两个body是否不会过小、是否不是薄面或棍'''
        def check_slice(dx, dy, dz, scale):
            for a, b in combinations([dx, dy, dz], 2):
                longer = max(a, b)
                shorter = min(a, b)
                if longer >= scale * shorter:
                    return True
            return False

        for min_p, max_p in zip(self.min_points, self.max_points):
            dx, dy, dz = max_p.x-min_p.x, max_p.y-min_p.y, max_p.z-min_p.z

            if dx < shortest or dy < shortest or dz < shortest:
                return False
            elif check_slice(dx,dy,dz, scale):
                return False
        return True


def test():
    # 薄片：00020002 长方体 ； 00020013 环; 00020244 复杂的形状；00020328两个圆柱； 00020845 有交集的形状； 00020352 远离的两个形状； 00020313 细长柱子
    # name = ['00020002', '00020013', '00020244', '00020328', '00020845', '00021807']
    name = ['00020328']
    b = []
    test_json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json/0002'
    for n in name:
        test_json_path = os.path.join(test_json_dir, n+'.json')
        a = Filter_Name_From_Shape(test_json_path)
        print(a.get_view())


def main():
    test()


if __name__ == "__main__":
    main()