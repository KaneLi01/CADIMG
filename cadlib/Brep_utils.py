import json
from copy import deepcopy
from .extrude import CADSequence
from .visualize import create_CAD, create_CAD_from_seq

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import breptools_OuterWire
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.GProp import GProp_GProps
from .macro import *


def get_BRep_from_file(file_path):
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        out_shape = create_CAD(cad_seq)
    except Exception as e:
        print("load and create failed.", e)

    return out_shape


def select_BRep_from_file(file_path, l, operation='add'):
    if operation == 'add':
        extrude_op = [0, 1]
    elif operation == 'sub':
        extrude_op = [2, 3]
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        base_seq = deepcopy(cad_seq.seq[:-1])
        operate_seq = deepcopy(cad_seq.seq[-1:])
        result_seq = deepcopy(cad_seq.seq)

        # 判断对象是否由大于2个body构成；且是并集操作
        if len(cad_seq.seq) == l and cad_seq.seq[l-1].operation in extrude_op:

            out_shape = {
                'base': create_CAD_from_seq(base_seq),
                'operate': create_CAD_from_seq(operate_seq)
            }

            # 判断body之间的距离是否过大
            dist = BRepExtrema_DistShapeShape(out_shape['base'], out_shape['operate'])
            if dist.IsDone() and dist.Value() < 1e-6:

                # 判断包围盒是否是相切的
                bbox1, _, _, corners1 = create_AABB_box(out_shape['base'])
                bbox2, _, _, corners2 = create_AABB_box(out_shape['operate'])
                body = BRepAlgoAPI_Common(bbox1, bbox2).Shape()
                props = GProp_GProps()
                brepgprop.VolumeProperties(body, props)
       
                if props.Mass() < 1e-5:

                    if corners2[0].X() - corners1[1].X() > -1e-4:
                        view = 'front'
                    elif corners1[0].X() - corners2[1].X() > -1e-4:
                        view = 'back'
                    elif corners2[0].Y() - corners1[1].Y() > -1e-4:
                        view = 'right'
                    elif corners1[0].Y() - corners2[1].Y() > -1e-4:
                        view = 'left'
                    elif corners2[0].Z() - corners1[1].Z() > -1e-4:
                        view = 'up'
                    elif corners1[0].Z() - corners2[1].Z() > -1e-4:    
                        view = 'down'
                    else: return None, None
                    

                    out_shape['result'] = create_CAD_from_seq(result_seq)
                    return out_shape, view
        return None, None
    except Exception as e:
        print("load and create failed.", e)
        return None, None


def get_points_from_BRep(shape):
    """
    提取 OpenCASCADE 几何体的所有顶点坐标
    :param shape: TopoDS_Shape
    :return: List of (x, y, z) 坐标
    """
    points = set()
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)  # 遍历顶点
    while explorer.More():
        vertex = explorer.Current()  # 获取当前顶点
        point = BRep_Tool.Pnt(vertex)  # 获取顶点的坐标
        points.add((point.X(), point.Y(), point.Z()))  # 转换为 (x, y, z) 格式
        explorer.Next()
    return points


def get_wireframe_from_body(shape):
    """
    通用方法：从 Shape/Compound 提取线框（Wire）
    支持以下输入类型：
        - 独立 Solid/Face/Wire/Edge
        - Compound（包含多个子几何体）
    """
    # Case 1: 如果本身就是 Wire，直接返回
    if shape.ShapeType() == TopAbs_WIRE:
        return TopoDS_Wire(shape)
    
    # Case 2: 处理 Compound 或包含多边的形状
    wire_builder = BRepBuilderAPI_MakeWire()
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    
    while explorer.More():
        # 正确转换方式：使用 topods_Edge() 函数
        edge = TopoDS_Wire(explorer.Current())
        wire_builder.Add(edge)
        explorer.Next()
    
    if wire_builder.IsDone():
        return wire_builder.Wire()
    
    # Case 3: 尝试从 Solid/Face 提取外轮廓
    try:
        return breptools_OuterWire(shape)
    except:
        pass
    
    # Case 4: 其他情况抛出异常
    raise RuntimeError("无法从该形状提取线框")


def get_wireframe(shape):
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        edge_list.append(edge)
        explorer.Next()

    return edge_list  # 返回生成的线框


def create_AABB_box(shape):
    try:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox, True)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        lower = gp_Pnt(xmin, ymin, zmin)
        upper = gp_Pnt(xmax, ymax, zmax)  
        center = (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2
        )
        ls = [xmax-xmin, ymax-ymin, zmax-zmin]

        corners = [lower, upper] 

        return BRepPrimAPI_MakeBox(lower, upper).Shape(), center, ls, corners
    
    except Exception as e:
        print("create result box failed.", e)    
        return None, None, None
