import json
import h5py
import random
import numpy as np
from OCC.Display.SimpleGui import init_display
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from cadlib.Brep_utils import get_BRep_from_file,  get_edges
import cadlib.Brep_utils as Brep_utils
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK, Quantity_NOC_RED
from OCC.Core.V3d import  V3d_DirectionalLight
from OCC.Core.Graphic3d import  Graphic3d_MaterialAspect
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.AIS import AIS_Shape, AIS_Line
from OCC.Core.gp import gp_Dir, gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.Prs3d import Prs3d_Drawer, Prs3d_LineAspect
from OCC.Core.Aspect import Aspect_TOL_DASH



# 定义全局颜色字典，使用 Quantity_Color 对象
COLORS = {
    "red": Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB),       # 红色
    "green": Quantity_Color(0.0, 1.0, 0.0, Quantity_TOC_RGB),     # 绿色
    "blue": Quantity_Color(0.0, 0.0, 1.0, Quantity_TOC_RGB),      # 蓝色
    "yellow": Quantity_Color(1.0, 1.0, 0.0, Quantity_TOC_RGB),    # 黄色
    "cyan": Quantity_Color(0.0, 1.0, 1.0, Quantity_TOC_RGB),      # 青色
    "magenta": Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB),   # 品红
    "white": Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB),     # 白色
    "black": Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB),     # 黑色

}


MECHANICAL_COLORS = {
    "copper":      (0.2294, 0.3314, 0.1510),     # 铜色
    "graphite":    (0.3412, 0.1294, 0.3020),       # 石墨色
    "stainless":   (0.5020, 0.5020, 0.4941),    # 不锈钢
    "blackish":    (0.2392, 0.2196, 0.2078),       # 偏黑
    "mahogany":    (0.5451, 0.2706, 0.0745),      # 马棕色
    "cobalt":      (0.2392, 0.3490, 0.6706),      # 钴蓝
}

from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_NOM_STONE, Graphic3d_NOM_OBSIDIAN, Graphic3d_NOM_COPPER, Graphic3d_NOM_PEWTER

MATERIALS = [
    Graphic3d_NOM_ALUMINIUM,
    Graphic3d_NOM_STONE,
    Graphic3d_NOM_OBSIDIAN,
    Graphic3d_NOM_COPPER,
    Graphic3d_NOM_PEWTER,
]



def get_mechanical_color(color_dict=MECHANICAL_COLORS):
    _, (r, g, b) = random.choice(list(color_dict.items()))
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def get_random_color(a1, a2):
    r, g, b = random.uniform(a1, a2), random.uniform(a1, a2), random.uniform(a1, a2)
    color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    return color


def get_color(r, g, b):
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def save_dash_wire(output_path, edge_list=None):
    """
    用于在线渲染形状或线框，或二者合并。用于调试。
    """

    if edge_list == None:
        raise Exception('input at least one item!')
    
    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create()  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
    # offscreen_renderer.SetPerspectiveProjection()
    offscreen_renderer.SetOrthographicProjection()
    offscreen_renderer.View.SetScale(1.0)

    # 设置随机浅色背景
    bg_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)

    # 设置线型
    drawer = Prs3d_Drawer()
    line_aspect = Prs3d_LineAspect(
        Quantity_Color(Quantity_NOC_BLACK),
        Aspect_TOL_DASH,
        3.0
    )
    drawer.SetWireAspect(line_aspect)

    if edge_list is not None:
        for edge in edge_list:
            ais_shape = AIS_Shape(edge)
            ais_shape.SetAttributes(drawer)
            offscreen_renderer.Context.Display(ais_shape, True)

    offscreen_renderer.FitAll()
    offscreen_renderer.View.Dump(output_path)


    

    # 设置摄像机
    # offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    # offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    # bg_color = get_random_color(bg_color, bg_color)
    # offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)






def display_BRep(shape=None, wire_list=None):
    """
    用于在线渲染形状或线框，或二者合并。用于调试。
    """

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    display, start_display, _, _ = init_display()

    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color,bg_color)

    # 设置摄像机
    display.View.SetEye(2, 2, 2)  
    display.View.SetAt(1, 1, 1)  
    display.View.SetScale(500)

    if wire_list is not None:
        for wire in wire_list:
            display.DisplayShape(wire, update=False, color="black")
    if shape is not None:
        ais_out_shape = display.DisplayShape(shape, update=False, color=get_mechanical_color(MECHANICAL_COLORS))
 
    start_display()


def display_BRep_with_bbox(shape=None):
    """
    用于在线渲染形状或线框，或二者合并。用于调试。
    """

    if shape == None:
        raise Exception('input at least one item!')

    display, start_display, _, _ = init_display()

    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color,bg_color)

    # 设置摄像机
    display.View.SetEye(2, 2, 2)  
    display.View.SetAt(1, 1, 1)  
    display.View.SetScale(500)
    
    ais_out_shape = display.DisplayShape(shape, update=False, color=get_mechanical_color(MECHANICAL_COLORS))

    bbox = Brep_utils.get_bbox(shape)
    bbox_shape = Brep_utils.create_box_from_minmax(bbox.min, bbox.max)
    bbox_edges = Brep_utils.get_edges(bbox_shape)

    for edge in bbox_edges:
        display.DisplayShape(edge, update=False, color="black")
 
    start_display()


def display_BRep_with_origin(shape=None, wire_list=None, show_origin=True, origin_radius=0.01, origin_color="red"):
    """
    用于在线渲染形状或线框，或二者合并。用于调试。
    额外显示原点球。
    
    参数:
    shape: 要显示的形状
    wire_list: 要显示的线框列表
    show_origin: 是否显示原点球 (默认True)
    origin_radius: 原点球半径 (默认0.01)
    origin_color: 原点球颜色 (默认红色)
    """

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    display, start_display, _, _ = init_display()

    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color, bg_color)

    # 设置摄像机
    display.View.SetEye(2, 2, 2)  
    display.View.SetAt(1, 1, 1)  
    display.View.SetScale(500)

    # 显示线框
    if wire_list is not None:
        for wire in wire_list:
            display.DisplayShape(wire, update=False, color="black")
    
    # 显示主形状
    if shape is not None:
        ais_out_shape = display.DisplayShape(shape, update=False, color=get_mechanical_color(MECHANICAL_COLORS))
    
    # 显示原点球
    if show_origin:
        # 创建原点处的球
        origin_point = gp_Pnt(0.0, 0.0, 0.0)
        sphere_maker = BRepPrimAPI_MakeSphere(origin_point, origin_radius)
        origin_sphere = sphere_maker.Shape()
        
        # 显示原点球
        display.DisplayShape(origin_sphere, update=False, color=origin_color)

    start_display()


def save_BRep(output_path, shape=None, wire_list=None, mode='orth',
              cam_pos=None, see_at=None, bg_color=1.0):
    '''离线渲染shape'''

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create()  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
   
    if mode == 'orth':
        offscreen_renderer.SetOrthographicProjection()
    elif mode == 'pers':
         offscreen_renderer.SetPerspectiveProjection()
    offscreen_renderer.View.SetScale(1.0)
    

    # 设置摄像机
    if cam_pos is not None:
        offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    if see_at is not None:
        offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    bg_color = get_random_color(bg_color, bg_color)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)

    if wire_list is not None:
        for wire in wire_list:
            ais_wire = AIS_Shape(wire)
            ais_wire.SetWidth(2.0)  # 设置线宽
            ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
            offscreen_renderer.Context.Display(ais_wire, False)
    
    if shape is not None:
        # 如果是true，会自动缩放。如果是False则不会
        offscreen_renderer.DisplayShape(shape, update=False, color=Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB))
    
    # offscreen_renderer.Repaint()
    offscreen_renderer.FitAll()
    # c = offscreen_renderer.GetCamera()
    # e = c.Eye()
    # ce = c.Center()
    # b = Brep_utils.get_bbox(shape)
    # print('bbox：',b.min, b.max, b.center)
    # print('cam：',e.X(),e.Y(),e.Z(), ce.X(),ce.Y(),ce.Z())

    # 试一下相机位姿 透视视图和blenderproc

    
    offscreen_renderer.View.Dump(output_path)


def save_BRep_with_white_faces(output_path, shape=None, wire_list=None, mode='orth',
              cam_pos=None, see_at=None, bg_color=1.0):
    '''离线渲染shape'''

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create()  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
   
    if mode == 'orth':
        offscreen_renderer.SetOrthographicProjection()
    elif mode == 'pers':
         offscreen_renderer.SetPerspectiveProjection()
    offscreen_renderer.View.SetScale(1.0)
    

    # 设置摄像机
    if cam_pos is not None:
        offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    if see_at is not None:
        offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    bg_color = get_random_color(bg_color, bg_color)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)


    
    if shape is not None:
        # 如果是true，会自动缩放。如果是False则不会
        offscreen_renderer.DisplayShape(shape, update=False, color=Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))
    if wire_list is not None:
        for wire in wire_list:
            ais_wire = AIS_Shape(wire)
            ais_wire.SetWidth(3.0)  # 设置线宽
            ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
            offscreen_renderer.Context.Display(ais_wire, True)
    
    offscreen_renderer.FitAll()
    
    offscreen_renderer.View.Dump(output_path)


def save_BRep_ortho_face_color(output_path, shape, cam_pos=[2,2,2], see_at=[0,0,0], bg_color=1.0):
    '''离线渲染shape，按照面来选择颜色'''
    import torch
    if isinstance(cam_pos, torch.Tensor):
            cam_pos = cam_pos.cpu().numpy().tolist()
    elif isinstance(cam_pos, np.ndarray):
            cam_pos = cam_pos.tolist()
        
    if isinstance(see_at, torch.Tensor):
            see_at = see_at.cpu().numpy().tolist()
    elif isinstance(see_at, np.ndarray):
            see_at = see_at.tolist()
    
    
    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create(phong_shading=False, create_default_lights=False)  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
    # offscreen_renderer.SetPerspectiveProjection()
    offscreen_renderer.SetOrthographicProjection()
    offscreen_renderer.View.SetScale(1.0)


    # 设置摄像机
    offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    bg_color = get_random_color(bg_color, bg_color)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)

    face_list = Brep_utils.get_faces(shape)
    wire_list = Brep_utils.get_edges(shape)

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    for wire in wire_list:
        ais_wire = AIS_Shape(wire)
        ais_wire.SetWidth(1.0)  # 设置线宽

        curve_adaptor = BRepAdaptor_Curve(wire)
        curve_type = curve_adaptor.GetType()
        if curve_type != 6:
            ais_wire.SetColor(Quantity_Color(Quantity_NOC_RED))
            offscreen_renderer.Context.Display(ais_wire, False)
        else:
            ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
            offscreen_renderer.Context.Display(ais_wire, False)             
        

    for i, face in enumerate(face_list):
        ais_face = AIS_Shape(face)
        c = (i+1)*5 / 255
        color = get_color(c,c,c)

        ais_face.SetColor(color)
        offscreen_renderer.Context.Display(ais_face, False)
    
    
    # if shape is not None:
    #     # 如果是true，会自动缩放。如果是False则不会
    #     offscreen_renderer.DisplayShape(shape, update=True, color=Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB))
    
    # offscreen_renderer.Repaint()
    offscreen_renderer.FitAll()
    # c = offscreen_renderer.GetCamera()
    # e = c.Eye()
    # ce = c.Center()
    # b = Brep_utils.get_bbox(shape)

    # 试一下相机位姿 透视视图和blenderproc
    offscreen_renderer.View.Dump(output_path)


def save_BRep_list(output_path, shape=None,
              cam_pos=[2,2,2], see_at=[0,0,0], bg_color=1.0):
    '''离线渲染shape；只渲染两个shape；第一shape为蓝色，第二shape为绿色'''

    if shape == None:
        raise Exception('input at least one item!')

    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create()  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
    #offscreen_renderer.SetPerspectiveProjection()
    offscreen_renderer.SetOrthographicProjection()
    # offscreen_renderer.View.SetScale(1.0)
    
    # 设置摄像机
    offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    bg_color = get_random_color(bg_color, bg_color)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)
    
    offscreen_renderer.DisplayShape(shape[0], update=False, color='blue')
    for wire in Brep_utils.get_edges(shape[0]):
        ais_wire = AIS_Shape(wire)
        ais_wire.SetWidth(1.0)  # 设置线宽
        ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
        offscreen_renderer.Context.Display(ais_wire, False)
    offscreen_renderer.DisplayShape(shape[1], update=False, color='green')
    for wire in Brep_utils.get_edges(shape[1]):
        ais_wire = AIS_Shape(wire)
        ais_wire.SetWidth(1.0)  # 设置线宽
        ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
        offscreen_renderer.Context.Display(ais_wire, False)
    
    offscreen_renderer.FitAll()
    
    offscreen_renderer.View.Dump(output_path)


def display_BRep_list_with_different_color(shapes=None, colors=None):
    """
    shapes: 单个shape或shape列表
    colors: 单个color或color列表，与shapes对应
    """
    
    if shapes is None:
        raise Exception('input at least one shape!')
    
    display, start_display, _, _ = init_display()
    
    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color, bg_color)
    
    # 设置摄像机
    display.View.SetEye(2, 2, 2)
    display.View.SetAt(1, 1, 1)
    display.View.SetScale(500)
    
    # 处理单个shape的情况，转换为列表
    if not isinstance(shapes, (list, tuple)):
        shapes = [shapes]
    
    # 处理颜色参数
    if colors is None:
        # 如果没有提供颜色，使用默认颜色或随机颜色
        colors = [get_mechanical_color(MECHANICAL_COLORS) for _ in shapes]
    elif not isinstance(colors, (list, tuple)):
        # 如果colors是单个颜色，应用到所有shapes
        colors = [colors] * len(shapes)
    else:
        # 如果colors列表长度不够，循环使用或用默认颜色填充
        if len(colors) < len(shapes):
            # 方法1: 循环使用现有颜色
            colors = colors * (len(shapes) // len(colors) + 1)
            colors = colors[:len(shapes)]
            
            # 方法2: 或者用默认颜色填充不足的部分
            # while len(colors) < len(shapes):
            #     colors.append(get_mechanical_color(MECHANICAL_COLORS))
    
    # 渲染每个shape及其对应颜色
    ais_shapes = []
    for i, shape in enumerate(shapes):
        color = colors[i] if i < len(colors) else get_mechanical_color(MECHANICAL_COLORS)
        ais_shape = display.DisplayShape(shape, update=False, color=color)
        ais_shapes.append(ais_shape)
    
    # 更新显示
    # display.FitAll()
    start_display()
    
    return ais_shapes


def display_BRep_bbox(shape):
    bbox = Brep_utils.get_bbox(shape)
    cube = Brep_utils.create_box_from_minmax(bbox.min, bbox.max)
    cube_wires = Brep_utils.get_edges(cube)
    display_BRep(shape, cube_wires)


class ARGS:
    def __init__(self):
        # self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0000/00000007.json"
        self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0002/00020718.json"
        self.save = False
        self.show_type = "body"  # wireframe | body



def main():

    output_path = '/home/lkh/siga/output/temp/ortho/1.png'
    step_path = '/home/lkh/siga/dataset/ABC/temp/step/00/00000029/00000029_ad34a3f60c4a4caa99646600_step_009.step'
    shape = Brep_utils.get_BRep_from_step(step_path)
    save_BRep_ortho_face_color(output_path=output_path, shape=shape)

if __name__ == '__main__':
    main()
    


