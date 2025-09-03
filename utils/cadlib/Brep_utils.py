import json, copy, os
import numpy as np
from .extrude import CADSequence
from .visualize import create_CAD, create_CAD_by_seq

import trimesh
from dataclasses import dataclass
from collections import namedtuple


from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Mat, gp_XYZ
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, TopoDS_Face, topods, topods_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepTools import breptools_OuterWire, breptools
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Geom import Geom_Plane, Geom_SphericalSurface
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from typing import List, Tuple

@dataclass
class Point:
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_list(cls, coords: List[float]) -> "Point":
        """Construct a Point from a list of 3 floats."""
        if len(coords) != 3:
            raise ValueError("Input list must contain exactly 3 elements (x, y, z).")
        return cls(coords[0], coords[1], coords[2])
    
@dataclass
class BBox:
    min: Point
    max: Point
    center: Point

    def lengths(self) -> tuple[float, float, float]:
        """计算边长"""
        dx = self.max.x - self.min.x
        dy = self.max.y - self.min.y
        dz = self.max.z - self.min.z
        return (dx, dy, dz)
    
    def max_length(self) -> float:
        """获取最长边"""
        return max(self.lengths())
    
    def to_shape(self):
        """将BBox转换为TopoDS_Solid长方体"""
        
        dx = self.max.x - self.min.x
        dy = self.max.y - self.min.y
        dz = self.max.z - self.min.z
        
        # 创建长方体
        box = BRepPrimAPI_MakeBox(
            gp_Pnt(self.min.x, self.min.y, self.min.z),
            dx, dy, dz
        ).Solid()
        
        return box
    
    def print_info(self):
        print(f"  最小点: {self.min}")
        print(f"  最大点: {self.max}")
        print(f"  中心点: {self.center}")


def get_BRep_from_seq(sub_seq):
    try:
        out_shape = create_CAD_by_seq(copy.deepcopy(sub_seq))
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


def get_BRep_from_file(file_path):
    try:
        cad_seq = get_seq_from_json(file_path)
        out_shape = get_BRep_from_seq(cad_seq.seq)
        return out_shape
    except Exception as e:
        print("load and create failed.", e)


def get_BRep_from_step(file_path):
    """读取STEP文件并返回TopoDS_Shape对象"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    
    if status != IFSelect_RetDone:
        return None
    
    reader.TransferRoots()  # 转换几何实体
    shape = reader.OneShape()  # 获取合并后的形状
    return shape


def save_Brep_to_step(shape: TopoDS_Shape, filename: str):
    """
    将一个 shape 保存为 step 文件
    :param shape: TopoDS_Shape 对象
    :param filename: 输出的 step 文件路径
    """
    writer = STEPControl_Writer()
    # 将shape加入writer
    writer.Transfer(shape, STEPControl_AsIs)
    # 写入文件
    status = writer.Write(filename)
    
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 文件写入失败，状态码: {status}")
    print(f"保存成功: {filename}")


def get_step_path_from_name(root_dir='/home/lkh/siga/dataset/ABC/step', name='00000000'):
    cls_idx = name[2:4]

    step_subdir = os.path.join(root_dir, cls_idx, name)
    step_file = os.listdir(step_subdir)
    step_path = os.path.join(step_subdir, step_file[0])
    return step_path


def get_bbox(shape, ld=0.01):
    all_verts, _, _ = shape2mesh(shape, linear_deflection=ld)

    xmin, ymin, zmin = all_verts.min(axis=0)
    xmax, ymax, zmax = all_verts.max(axis=0)
    min=Point(xmin, ymin, zmin)
    max=Point(xmax, ymax, zmax)
    center=Point((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)

    return BBox(min, max, center)


# 该方法有误差
# def get_bbox(shape: TopoDS_Compound):
#     """获取shape的包围盒"""
#     bbox = Bnd_Box()
#     # mesh = BRepMesh_IncrementalMesh(shape, 0.1)
#     # mesh.Perform()
#     brepbndlib.Add(shape, bbox, False)

#     xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
#     min=Point(xmin, ymin, zmin)
#     max=Point(xmax, ymax, zmax)
#     center=Point((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)

#     return BBox(min, max, center)


def print_bbox_info(shape: TopoDS_Compound):
    """打印shape的包围盒信息"""
    bbox = get_bbox(shape)
    p_min = bbox.min
    p_max = bbox.max
    xmin, ymin, zmin = p_min.x, p_min.y, p_min.z
    xmax, ymax, zmax = p_max.x, p_max.y, p_max.z
    center = bbox.center.to_list()
    length = xmax - xmin  
    width = ymax - ymin   
    height = zmax - zmin 
    
    # 计算各面面积
    front_area = width * height   
    top_area = length * width    
    right_area = length * height  
    
    # 打印信息
    print("="*50)
    print("包围盒信息:")
    print("="*50)
    
    print(f"最小点坐标: ({xmin:.3f}, {ymin:.3f}, {zmin:.3f})")
    print(f"最大点坐标: ({xmax:.3f}, {ymax:.3f}, {zmax:.3f})")
    print()
    
    print(f"长度: {length:.3f}")
    print(f"宽度: {width:.3f}")
    print(f"高度: {height:.3f}")
    print()
    
    print(f"中心点坐标: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print()
    
    print("各面面积:")
    print(f"  前面: {front_area:.3f}")
    print(f"  上面: {top_area:.3f}")
    print(f"  右面: {right_area:.3f}")
    print("="*50)


def get_longest_edge_of_bbox(shape):
    """获取shape包围盒的最长边长度"""
    # 创建并计算包围盒
    bbox = get_bbox(shape)
    
    # 返回最大的边长
    return bbox.max_length()


def get_scale_of_bbox(shape):
    """获取shape包围盒的所有边长度的总和"""
    # 创建并计算包围盒
    bbox = get_bbox(shape)
    
    # 返回所有边之和
    return sum(bbox.lengths())


def get_vertices(shape):
    """获取shape的顶点"""
    vertices_list = set()  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)  # 遍历 Compound 中的边
    while explorer.More():
        point = explorer.Current()  # 获取当前边 
        vertices_list.add(point)
        explorer.Next()

    return vertices_list  # 返回生成的线框


def get_edge_veritces(edge):
    vertices = []
    explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while explorer.More():
        vertex = topods.Vertex(explorer.Current())
        vertices.append(BRep_Tool.Pnt(vertex))
        explorer.Next()
    
    # 返回起点和终点
    return vertices[0], vertices[-1]


def get_edges(shape):
    """获取shape的线框"""
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        edge_list.append(edge)
        explorer.Next()

    return edge_list  # 返回生成的线框


def get_faces(shape):
    """获取shape的面"""
    faces = set()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()  # 获取当前面（可选）
        faces.add(face)
        explorer.Next()
    return list(faces)

def get_edges_from_face(face):
    edges = set()
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        edges.add(edge)
        edge_explorer.Next()
    
    return list(edges)

def get_volume(shape):
    """计算shape的体积"""
    if shape is None or shape.IsNull():
        return 0.0

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def get_min_distance(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> float:
    """计算两个Shape之间的最小距离"""
    dist_calc = BRepExtrema_DistShapeShape(shape1, shape2)
    dist_calc.Perform()
    if not dist_calc.IsDone():
        raise RuntimeError("距离计算失败")
    return dist_calc.Value()


def get_common_volume(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> float:
    common = BRepAlgoAPI_Common(shape1, shape2)
    common_shape = common.Shape()
    common_volume = get_volume(common_shape)

    return common_volume


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

    return edge_list 


def create_box_from_minmax(min_point, max_point):
    """
    从最小点和最大点创建平行于坐标轴的topoDS长方体
    """
    if isinstance(min_point, Point):
        min_point = min_point.to_list()
    if isinstance(max_point, Point):
        max_point = max_point.to_list()        

    dx = max_point[0] - min_point[0]
    dy = max_point[1] - min_point[1]
    dz = max_point[2] - min_point[2]

    # 创建长方体
    box = BRepPrimAPI_MakeBox(
        gp_Pnt(min_point[0], min_point[1], min_point[2]),  # 起点（最小点）
        dx, dy, dz
    ).Solid()

    return box


def test_create_2boxs(mode='intersection'):
    """
    创建不同的box实例，用于测试
    """
    box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Solid()

    if mode == 'inter':
        p2 = gp_Pnt(0.5, 0.5, 0)
    elif mode == 'tang':
        p2 = gp_Pnt(1, 0, 0)
    elif mode == 'away':
        p2 = gp_Pnt(2, 2, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    box2 = BRepPrimAPI_MakeBox(p2, 1, 1, 1).Solid()
    return box1, box2
    

def explore_shape(shape, level=0):
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer


    """仅打印直接子层级（非递归）"""
    shape_type = shape.ShapeType()
    
    # 类型映射
    type_names = {
        TopAbs_COMPOUND: "COMPOUND",
        TopAbs_SOLID: "SOLID",
        TopAbs_SHELL: "SHELL",
        TopAbs_FACE: "FACE"
    }
    
    print(f"父形状: {type_names.get(shape_type, 'UNKNOWN')} (子级数量: {shape.NbChildren()})")
    
    # 初始化探索器（遍历所有直接子级）
    explorer = TopExp_Explorer()
    explorer.Init(shape, TopAbs_COMPOUND)  # 可替换为TopAbs_SOLID等
    
    childs = []

    # 仅遍历直接子级
    while explorer.More():
        child = explorer.Current()
        child_type = child.ShapeType()
        print(f"  └─ {type_names.get(child_type, 'UNKNOWN')} (子级数量: {child.NbChildren()})")
        childs.append(child)
        explorer.Next()
    return childs

def get_face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass() 
    return area

def get_faces_area_type(shape):
    '''plane, cylinder, cone, sphere, torus, beziersurface, bsplinesurface, surfaceofrevolution, surfaceofextrusion, othersurface'''
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAdaptor import GeomAdaptor_Surface

    area_list = []
    type_list = []

    face_list = get_faces(shape)
    for face in face_list:
        area_list.append(get_face_area(face))

        surface = BRep_Tool.Surface(face)
        adaptor = GeomAdaptor_Surface(surface)
        surf_type = adaptor.GetType()  # 返回 GeomAbs_SurfaceType 枚举
        type_list.append(surf_type)

    return area_list, type_list


def get_edges_len_type(shape):
    '''line, circle, ellipse, hyperbola, parabola, beziercurve, bsplinecurve, othercurve.'''
    from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
    len_list = []
    type_list = []
    
    edge_list = get_edges(shape)
    for edge in edge_list:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()
        len_list.append(length)

        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()
        type_list.append(curve_type)

    return len_list, type_list


def get_child_shapes(shape):
    from OCC.Core.TopoDS import TopoDS_Iterator, TopoDS_Shape
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
    it = TopoDS_Iterator(shape) 
    count = 0

    childs = []
    while it.More():  
        child = it.Value()  
        count += 1

        childs.append(child)
        it.Next()  

    return childs


def shape2mesh(shape, linear_deflection=1e-4):
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesh.Perform()
    
    all_verts = []
    all_faces = []
    vert_offset = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            exp.Next()
            continue

        nb_nodes = triangulation.NbNodes()
        points = [triangulation.Node(i + 1) for i in range(nb_nodes)]
        triangles = triangulation.Triangles()

        # 添加顶点
        for p in points:
            all_verts.append([p.X(), p.Y(), p.Z()])

        # 添加面（注意 OCC 索引从 1 开始）
        for i in range(triangulation.NbTriangles()):
            tri = triangles.Value(i + 1)
            n1, n2, n3 = tri.Get()
            all_faces.append([
                n1 - 1 + vert_offset,
                n2 - 1 + vert_offset,
                n3 - 1 + vert_offset
            ])

        vert_offset += nb_nodes
        exp.Next()

    # 转换为 numpy 数组
    all_verts = np.array(all_verts)
    all_faces = np.array(all_faces)
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    return all_verts, all_faces, mesh


def shape2mesh_save(shape, output_path, linear_deflection=1e-4):
    _, _, mesh = shape2mesh(shape, linear_deflection=1e-4)
    mesh.merge_vertices()  # 合并重复顶点
    mesh.remove_duplicate_faces()  # 移除重复面
    '''将shape网格化并保存到指定路径'''
    # 进行网格化
    # mesh = BRepMesh_IncrementalMesh(shape, linear_deflection)

    print(f"零面积三角形: {np.sum(mesh.area_faces < 1e-10)}")
    # 确保法向量朝外
    if not mesh.is_watertight:
        print("警告: 网格不是封闭的，法向量方向可能不准确")
    mesh.fix_normals()

    # if mesh.volume < 0:
    #     mesh.invert()

    mesh.export(output_path)

def sample_edge(edge, n_points=20):
    """
    对单条边均匀采样，返回采样点列表[(x,y,z), ...]
    """
    adaptor = BRepAdaptor_Curve(edge)
    discretizer = GCPnts_UniformAbscissa(adaptor, n_points)
    points = []
    if not discretizer.IsDone():
        # 采样失败，取端点
        p1 = adaptor.Value(adaptor.FirstParameter())
        p2 = adaptor.Value(adaptor.LastParameter())
        points = [(p1.X(), p1.Y(), p1.Z()), (p2.X(), p2.Y(), p2.Z())]
    else:
        for i in range(1, discretizer.NbPoints() + 1):
            p = adaptor.Value(discretizer.Parameter(i))
            points.append((p.X(), p.Y(), p.Z()))
    return points


def normalize_shape(shape, target_len=1.5):
    # 计算包围盒
    bbox = get_bbox(shape)
    
    # 当前长宽高
    max_len = bbox.max_length()
    center = bbox.center
    
    # 先平移到原点（包围盒中心对齐原点）
    cx, cy, cz = center.x, center.y, center.z

    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()
    
    # 计算缩放比例
    scale_factor = target_len / max_len
    
    # 缩放
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), scale_factor)  # 以原点为缩放中心
    shape_scaled = BRepBuilderAPI_Transform(shape_centered, trsf_scale, True).Shape()
    
    return shape_scaled


def trans_to_ori(shape):
    # 计算包围盒
    bbox = get_bbox(shape, ld=0.1)
    
    # 当前长宽高
    center = bbox.center
    
    # 先平移到原点（包围盒中心对齐原点）
    cx, cy, cz = center.x, center.y, center.z

    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()
    
    return shape_centered    


def make_compound(shape_list):
    merged_shape = shape_list[0]
    for i, shape in enumerate(shape_list):
        if get_volume(shape) > 0.0:
            merged_shape = shape
            break
    for shape in shape_list[i+1:]:
        if get_volume(shape) <= 0.0:
            continue
        merged_shape = BRepAlgoAPI_Fuse(merged_shape, shape).Shape()

    return merged_shape


def merge_faces_compound(face_list):
    """
    将face列表合并为一个Compound（最简单的方式）
    """
    if not face_list:
        raise ValueError("face_list不能为空")
    
    # 创建复合体
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    
    # 将每个face添加到复合体中
    for face in face_list:
        builder.Add(compound, face)
    
    return compound


def merge_shapes(shape_list):
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for shape in shape_list:
        builder.Add(compound, shape)

    return compound


def is_planes_equal(face1, face2, ang_tol=1e-2, dist_tol=1e-3):
    from OCC.Core.GeomAbs import GeomAbs_Plane
    # 获取平面 1
    surf1 = BRepAdaptor_Surface(face1)
    if surf1.GetType() != 0:  # 0 = GeomAbs_Plane
        return False
    plane1 = BRepAdaptor_Surface(face1)
    if plane1.GetType() != GeomAbs_Plane:
        return False
    gp_plane1 = plane1.Plane()

    # 获取平面 2
    surf2 = BRepAdaptor_Surface(face2)
    if surf2.GetType() != 0:
        return False
    plane2 = BRepAdaptor_Surface(face2)
    if plane2.GetType() != GeomAbs_Plane:
        return False
    gp_plane2 = plane2.Plane()

    # 1. 检查法向是否平行
    dir1 = gp_plane1.Axis().Direction()
    dir2 = gp_plane2.Axis().Direction()
    if not dir1.IsParallel(dir2, ang_tol):
        return False

    # 2. 检查平面间距离
    pnt1 = gp_plane1.Location()  # 平面 1 上的一点
    dist = abs(gp_plane2.Distance(pnt1))  # pnt1 到平面 2 的距离
    if dist > dist_tol:
        return False

    return True


def get_valid_biggest_plane(shape, thre=1/4):
    # 输入shape，返回最大的平面和对应的包围盒平面；如果没有平面，则返回(None, 最大包围盒平面)
    face_area_list, face_type_list = get_faces_area_type(shape)  # 获取所有面的面积和类型，用于选择和排序

    bbox = get_bbox(shape)
    bbox_shape = bbox.to_shape()
    bbox_faces = get_faces(bbox_shape)
    bbox_face_area_list, _ = get_faces_area_type(bbox_shape)
    biggest_bbox_face_area = max(bbox_face_area_list)

    if 0 in face_type_list:
        # 如果有平面，则选择面积最大的平面，且需要和包围盒相切
        face_list = get_faces(shape)

        valid_faces = []
        valid_faces_area = []

        # 筛选面积大于阈值的面
        for face, area, t in zip(face_list, face_area_list, face_type_list):
            if t == 0:
                if area / biggest_bbox_face_area > thre:
                    valid_faces.append(face)
                    valid_faces_area.append(area)
        # 按照area大到小排列
        sorted_faces = [face for face_area, face in sorted(zip(valid_faces_area, valid_faces), key=lambda x: x[0], reverse=True)]
        # 找到最大的平面和对应包围盒平面
        # 可能遍历所有的平面都没有return，那么进入下一个步骤
        for face in sorted_faces:
            for bface in bbox_faces:
                if is_planes_equal(face, bface):
                    return (face, bface)  

    bbox_biggest_face_idx = bbox_face_area_list.index(biggest_bbox_face_area)
    return (None, bbox_faces[bbox_biggest_face_idx])  # 如果没有平面，则返回包围盒的最大平面


def numpy_to_trsf(H):
    """把4x4 numpy矩阵转成 gp_Trsf"""
    assert H.shape == (4, 4)

    # 提取旋转和平移
    R = H[:3, :3]
    t = H[:3, 3]

    # gp_Mat: 按列写入
    mat = gp_Mat(R[0,0], R[0,1], R[0,2],
                 R[1,0], R[1,1], R[1,2],
                 R[2,0], R[2,1], R[2,2])
    vec = gp_XYZ(t[0], t[1], t[2])

    return mat, vec


def apply_transformation(shape, H, copy=True):
    mat, vec = numpy_to_trsf(H)
    trsf_r = gp_Trsf()
    trsf_r.SetRotation(mat)
    shape_r = BRepBuilderAPI_Transform(shape, trsf_r, True).Shape()

    trsf_t = gp_Trsf()
    trsf_t.SetScale(vec)  # 以原点为缩放中心
    shape_t = BRepBuilderAPI_Transform(shape_r, trsf_t, True).Shape()    

    return shape_t

    trsf_translate = gp_Trsf()
    trsf_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, trsf_translate, True).Shape()
    
    # 计算缩放比例
    scale_factor = target_len / max_len
    
    # 缩放
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), scale_factor)  # 以原点为缩放中心
    shape_scaled = BRepBuilderAPI_Transform(shape_centered, trsf_scale, True).Shape()

def get_sphere_paras_from_face(face):
    """
    输入: face (TopoDS_Face)，已知它是球面
    输出: gp_Pnt 球心
    """
    # 从面得到几何曲面
    surface = BRep_Tool.Surface(face)
    
    # 尝试 downcast 成球面
    spherical_surface = Geom_SphericalSurface.DownCast(surface)
    
    # 获取球心
    center = spherical_surface.Location()
    radius = spherical_surface.Radius()
    return center, radius

def point_to_shape_distance(pnt: gp_Pnt, shape) -> float:
    """
    计算 gp_Pnt 到 TopoDS_Shape 的最小距离
    """
    # 点转成 vertex
    vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
    
    # 计算最小距离
    dist_calc = BRepExtrema_DistShapeShape(vertex, shape)
    dist_calc.Perform()
    
    if not dist_calc.IsDone():
        raise RuntimeError("最小距离计算失败")
    
    return dist_calc.Value()