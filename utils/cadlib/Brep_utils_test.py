import json
from .extrude import CADSequence
from .visualize_test import create_CAD, create_CAD_by_seq

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, TopoDS_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import breptools_OuterWire
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def get_BRep_from_seq(sub_seq):
    try:
        out_shape = create_CAD_by_seq(sub_seq)
        return out_shape
    except Exception as e:
        print("create shape from cad_seq fail.", e)


def get_seq_from_json(file_path):
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        return cad_seq
    except Exception as e:
        print("read json or get seq failed.", e)



def get_faces_from_BRep(shape):
    faces = set()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()  # 获取当前面（可选）
        faces.add(face)
        explorer.Next()
    return faces



def get_wireframe(shape):
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        edge_list.append(edge)
        explorer.Next()

    return edge_list  # 返回生成的线框



def get_wireframe_cir(shape):
    from OCC.Core.Geom import Geom_Line
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        

        curve_handle, _, _ = BRep_Tool.Curve(edge)
        if curve_handle.DynamicType().Name() == "Geom_Circle":
            edge_list.append(edge)


        explorer.Next()

    return edge_list  # 返回生成的线框



def get_bbox(shape: TopoDS_Compound):
    # 获取shape的包围盒坐标
    bbox = Bnd_Box()

    brepbndlib.Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    
    return (xmin, ymin, zmin, xmax, ymax, zmax)

