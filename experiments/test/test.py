import sys, datetime, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.cadlib import Brep_utils
from utils.vis import render_cad
from utils.rootdir_processer import FileProcessor
import utils.jsonl_utils as jsonl_utils


from OCC.Core.BRep import BRep_Tool


# def triangulate_shape(shape, deflection=0.1):



def write_obj(filename, verts, tris):
    with open(filename, 'w') as f:
        # 写入顶点
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 写入面（OBJ 的索引从 1 开始）
        for tri in tris:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def process_directory(root_dir, output_dir="output"):
    """处理目录下的所有 OBJ 文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    obj_files = []
    # 遍历所有子目录，收集 OBJ 文件路径
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(subdir, file))

    print(obj_files[4899])
    raise Exception('-')
    
    # 每 10 个 OBJ 文件为一组
    for i in range(0, len(obj_files), 10):
        group = obj_files[i:i+10]
        img_paths = []
        # 渲染每组 OBJ 文件
        for j, obj_path in enumerate(group):
            img_path = os.path.join(output_dir, f"temp_{i+j}.png")
            if render_obj_to_image(obj_path, img_path):
                img_paths.append(img_path)
        # 合并图片
        if img_paths:
            combined_path = os.path.join(output_dir, f"combined_{i//10}.png")
            combine_images(img_paths, combined_path)
            print(f"Saved combined image: {combined_path}")
            # 清理临时文件
            for img_path in img_paths:
                os.remove(img_path)



def get_num(n):
    
    root_dir = "/home/lkh/siga/dataset/ABC/abc_obj/00"  # 替换为你的 OBJ 文件根目录
    
    obj_files = []
    # 遍历所有子目录，收集 OBJ 文件路径
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(subdir, file))

    print(obj_files[n])
    return obj_files[n]


def look(n):
    import copy
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    file1 = get_num(n=n).replace('obj', 'step').replace('trimesh', 'step')


    shape = copy.deepcopy(Brep_utils.get_BRep_from_step(file1))
    l = Brep_utils.get_first_level_shapes(shape)
    mesh = BRepMesh_IncrementalMesh(l[1], 0.0001)
    mesh.Perform()
    stl_writer = StlAPI_Writer()
    stl_path = "/home/lkh/siga/output/temp/2256_1.stl"
    stl_writer.Write(shape, stl_path)

    mesh = trimesh.load(stl_path)
    mesh.export(stl_path.replace('stl', 'obj')) 

    # render_cad.display_BRep(shape)
    # Brep_utils.explore_shape(shape)
    # l = Brep_utils.get_first_level_shapes(shape)
    # # b1 = Brep_utils.get_bbox(l[0])
    # # b2 = Brep_utils.get_bbox(l[1])
    # # print(b1, b2)
    # # print(l)
    # for ll in l:
    #     render_cad.display_BRep(ll)
    # render_cad.save_BRep(output_path=o1, shape=l[0], see_at=[-198.69615714153213,1.4759205120735075,11.244403690310648], cam_pos=[-150.9623117688493,71.209765884756315,60.97824906299346])
    # render_cad.save_BRep(output_path=o2, shape=l[1], see_at=[0.6146319062223426,0.13443163012419745,18.638946488677888], cam_pos=[56.22102892490321,77.74082864880506,76.24534350735875])


def create_sphere_meshes(points, radius=0.02, subdivisions=3):
    """将所有点变为球体网格"""
    spheres = []
    for pt in points:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
        sphere.apply_translation(pt)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)

def make_cylinder(p1, p2, radius=0.1, sections=16):
    """创建两个点之间的圆柱体（三角网格）"""
    vec = np.array(p2) - np.array(p1)
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return None

    direction = vec / height
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cylinder.apply_translation([0, 0, height / 2.0])

    axis = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(axis)
    cylinder.apply_translation(p1)
    return cylinder




def convert_edges_to_mesh(edges, line_radius=0.05):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    """将STEP中的边转换为mesh网格：直线为圆柱，弧线为环面"""
    all_meshes = []
    for edge in edges:
        curve_adapt = BRepAdaptor_Curve(edge)
        curve_type = curve_adapt.GetType()
        print(f"处理曲线类型: {curve_type}")
        first = curve_adapt.FirstParameter()
        last = curve_adapt.LastParameter()

        p1 = curve_adapt.Value(first)
        p2 = curve_adapt.Value(last)

        pt1 = np.array([p1.X(), p1.Y(), p1.Z()])

        pt2 = np.array([p2.X(), p2.Y(), p2.Z()])

        if curve_type == 0:  # Line
            cyl = make_cylinder(pt1, pt2, radius=line_radius)
            cyl.apply_translation([0, 0, 0.5])
            if cyl:
                all_meshes.append(cyl)
        elif curve_type == 8:  # Circle (即弧线)
            circ = curve_adapt.Circle()
            center = circ.Location()
            axis = circ.Axis().Direction()
            torus = make_torus(
                center=[center.X(), center.Y(), center.Z()],
                normal=[axis.X(), axis.Y(), axis.Z()],
                radius=circ.Radius(),
                tube_radius=line_radius
            )
            if torus:
                all_meshes.append(torus)
        else:
            print(f"跳过未处理的曲线类型: {curve_type}")
    return trimesh.util.concatenate(all_meshes)




def render_yuan():
    shape = Brep_utils.get_BRep_from_step('/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step/add/00000257.step')
    Brep_utils.print_bbox_info(shape)
    faces = Brep_utils.get_faces(shape)
    
    c, r = Brep_utils.get_sphere_paras_from_face(faces[0])
    print(c.X(), c.Y(), c.Z())


    b = Brep_utils.create_box_from_minmax([0.0, 0.0, 0.0], [-1.0, 2.0, 2.0])
    print(Brep_utils.point_to_shape_distance(c, b))
    


def test_shape():
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Display.OCCViewer import Viewer3d
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK

    step_path1 = '/home/lkh/siga/dataset/ABC/new/c1/base_add/1/step/add/00000006.step'
    step_path2 = '/home/lkh/siga/dataset/ABC/step/44/00441808/00441808_21f1e2b653dba8d838a2ab87_step_000.step'
    s1 = Brep_utils.get_BRep_from_step(step_path1)
    s2 = Brep_utils.get_BRep_from_step(step_path2)
    renderer = Viewer3d()

    renderer.Create()  

    renderer.SetModeShaded()
    renderer.SetSize(512, 512)
    renderer.View.SetScale(500)

    renderer.SetOrthographicProjection()
    color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
    renderer.View.SetBgGradientColors(color, color)

    black_color = Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)
    # renderer.DisplayShape(s1, update=True, color=black_color)
    renderer.DisplayShape(s1, update=False, color=black_color)

    output_path1 = '/home/lkh/siga/CADIMG/experiments/test/1.png'
    renderer.View.Dump(output_path1)
    


def test_processer():
    from typing import List, Dict, Tuple, Optional
    class MyProcessor(FileProcessor):
        def __init__(self, root_dir, extension):
            super().__init__(root_dir, extension=extension)
            self.name_list = []

        def process_file(self, input_filepath: str):
            name = input_filepath.split('/')[-1].split('_')[0]
            self.name_list.append(name)

    # 实例化并使用
    processor = MyProcessor(
        root_dir="/home/lkh/siga/dataset/ABC/step",
        extension=".step",
    )

    # 处理文件
    processor.process_all()    
    print(processor.name_list)


def test_read_data():
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/0824/2/add.jsonl'
    add_feats = jsonl_utils.load_jsonl_to_list(jsonl_path)
    cones = 0
    spheres = 0
    toruses = 0
    total  = 0
    for feat in add_feats:
        # t = feat['faces_type']
        cn = feat['face_num']

        if cn[0] <= 16:
            total += 1
            '''plane, cylinder, cone, sphere, torus'''
            t = feat['faces_type']
            if 2 in t:
                cones += 1
            if 3 in t:
                spheres += 1
            if 4 in t:
                toruses += 1
    print(f"cones: {cones}, spheres: {spheres}, toruses: {toruses}, total: {total}")

def npy_to_mask_image(npy_path, save_path=None):
    """
    将 npy 文件中的 0/1 数组转换成黑白图并保存
    1 -> 黑色, 0 -> 白色

    Args:
        npy_path (str): 输入的 .npy 文件路径
        save_path (str, optional): 输出图像路径（默认同目录同名 .png）
    """
    # 读取数组
    mask = np.load(npy_path)

    # 检查是否为0/1数组
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 1 -> 0 (黑色), 0 -> 255 (白色)
    img_array = np.where(mask == 1, 0, 255).astype(np.uint8)

    # 转换成PIL图像
    img = Image.fromarray(img_array, mode="L")

    # 默认保存路径
    if save_path is None:
        save_path = os.path.splitext(npy_path)[0] + ".png"

    img.save(save_path)
    print(f"保存成功: {save_path}")


def main():

    npy_to_mask_image(npy_path='/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/normal_add/mask/00000006/00000006_0.png.npy', save_path='/home/lkh/siga/CADIMG/experiments/test/1.png')




if __name__ == "__main__":
    main()

