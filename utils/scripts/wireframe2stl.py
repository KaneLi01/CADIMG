from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.TopoDS import topods
import numpy as np
import trimesh




def extract_wireframe(step_path, n_points_per_edge=20):
    """
    从STEP文件中提取线框，将每条边均匀采样重构为直线段
    返回格式：list of line segments, 每条线段为 [(x1,y1,z1), (x2,y2,z2)]
    """
    # 读取STEP文件
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError("STEP文件读取失败")

    reader.TransferRoots()
    shape = reader.OneShape()

    exp = TopExp_Explorer(shape, TopAbs_EDGE)

    lines = []

    while exp.More():
        edge = topods.Edge(exp.Current())
        points = sample_edge(edge, n_points_per_edge)
        # 将采样点两两连接成线段
        for i in range(len(points) - 1):
            lines.append([points[i], points[i+1]])
        exp.Next()

    return lines




if __name__ == "__main__":
    step_file = "/home/lkh/siga/CADIMG/experiments/test/00200000/00200000_38fdf5917c579929b681cd30_step_003.step"  # 修改为你的STEP文件路径
    lines = extract_wireframe(step_file, n_points_per_edge=50)
    meshes = []
    for p1, p2 in lines:
        cyl = make_cylinder(p1, p2, radius=0.4)  # 你可以调节圆柱半径
        if cyl is not None:
            meshes.append(cyl)

    # 合并所有圆柱网格
    combined_mesh = trimesh.util.concatenate(meshes)

    # 保存为文件，例如STL或OBJ
    combined_mesh.export("/home/lkh/siga/CADIMG/experiments/test/00200000/00200000.stl")
