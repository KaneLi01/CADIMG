import json
from .extrude import CADSequence
from .visualize import create_CAD

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import breptools_OuterWire


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
