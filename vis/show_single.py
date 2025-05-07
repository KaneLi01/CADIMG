import json
import h5py
import random
import numpy as np
from OCC.Display.SimpleGui import init_display
import sys

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD, get_wireframe_from_body
from cadlib.Brep_utils import get_BRep_from_file, get_points_from_BRep, get_wireframe
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.V3d import  V3d_DirectionalLight
from OCC.Core.Graphic3d import  Graphic3d_MaterialAspect
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.AIS import AIS_Shape, AIS_Line
from OCC.Core.gp import gp_Dir



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


# 要把摄像机位置、shape颜色等随机信息作为参数传入，写一个新函数
def save_BRep_img(shape, output_path=None, seed=42):
    random.seed(seed)  # 确保每个数据的初始和修改后相同
    
    offscreen_renderer = Viewer3d()  # 离线渲染
    offscreen_renderer.Create()  # 初始化
    # 设置渲染模式为阴影模式，显示面信息
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)

    # 给物品添加材质
    ais_shape = AIS_Shape(shape)
    random_material = random.choice(MATERIALS)
    material = Graphic3d_MaterialAspect(random_material)
    offscreen_renderer.Context.SetMaterial(ais_shape, material, True)
    shape_color = get_random_color(0.05, 0.15)
    offscreen_renderer.Context.SetColor(ais_shape, shape_color, True)
    offscreen_renderer.Context.Display(ais_shape, True)

    # 添加光源
    light_color = get_random_color(0.9, 1.0)  
    # 随机方向
    random_dir = gp_Dir(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    light = V3d_DirectionalLight(random_dir, light_color, True)  # 创建光源
    light.SetIntensity(1.0)  # 设置光强
    offscreen_renderer.Viewer.AddLight(light)  # 将光源添加到渲染器
    offscreen_renderer.Viewer.SetLightOn()  # 打开光源

    # 设置摄像机
    random.seed(seed)
    x_eye = random.uniform(1.1, 1.5)
    y_eye = random.uniform(0.6, 0.8)
    z_eye = random.uniform(0.6, 0.8)
    offscreen_renderer.View.SetEye(x_eye, y_eye, z_eye)  # 设置摄像机位置
    offscreen_renderer.View.SetAt(0.1, 0.3, 0.3) 
    offscreen_renderer.View.SetScale(400)

    # 设置背景颜色
    bg_color = get_random_color(0.75, 1.0)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)


    # 绘制线框为黑色（保证离散形状的线框）
    edge_list = get_wireframe(shape) 
    for edge in edge_list:
        offscreen_renderer.DisplayShape(edge, update=False, color="black")


    offscreen_renderer.Repaint()
    offscreen_renderer.View.Dump(output_path)


def save_BRep_wire_img(wire_list, output_path=None, seed=42):
    random.seed(seed)  # 确保每个数据的初始和修改后相同
    
    offscreen_renderer = Viewer3d()  # 离线渲染
    offscreen_renderer.Create()  # 初始化
    # 设置渲染模式为阴影模式，显示面信息
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)

    # 设置摄像机
    random.seed(seed)
    x_eye = random.uniform(1.1, 1.5)
    y_eye = random.uniform(0.6, 0.8)
    z_eye = random.uniform(0.6, 0.8)
    offscreen_renderer.View.SetEye(x_eye, y_eye, z_eye)  # 设置摄像机位置
    offscreen_renderer.View.SetAt(0.1, 0.3, 0.3) 
    offscreen_renderer.View.SetScale(400)

    # 设置背景颜色
    bg_color = get_random_color(1.0, 1.0)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)

    # 绘制线框为黑色（保证离散形状的线框）
    for edge in wire_list:
        ais_edge = AIS_Shape(edge)
        ais_edge.SetWidth(5.0)  # 设置线宽
        ais_edge.SetColor(Quantity_Color(Quantity_NOC_BLACK))
        offscreen_renderer.Context.Display(ais_edge, False)

    offscreen_renderer.Repaint()
    offscreen_renderer.View.Dump(output_path)


# 用save_BRep_img保存，这个用于交互式渲染
def show_BRep(out_shape, show_type, display=None, save_path=None):
    if display is None:
        display, start_display, _, _ = init_display()
        close_display = True
    else:
        close_display = False

    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color,bg_color)
    # display.View.SetBackgroundColor(bg_color)


    # 设置摄像机
    display.View.SetEye(1, 1, 1)  
    display.View.SetAt(1, 1, 1)  
    display.View.SetScale(500)

    edge_list = get_wireframe(out_shape)  # 将形状的边提取出来，防止离散的形状
    # 绘制边
    for edge in edge_list:
        display.DisplayShape(edge, update=False, color="black")

    if show_type == "body":
        ais_out_shape = display.DisplayShape(out_shape, update=False, color=get_mechanical_color(MECHANICAL_COLORS))

    display.View.TriedronErase()  # 清除坐标轴

    if save_path is not None:
        # 保存
        display.Repaint()
        display.View.Dump(save_path)
        display.EraseAll()  # 清除显示内容
        if close_display:
            display.Context.Delete()  # 完全关闭显示上下文
    else:    
        start_display()


def run(args):
    out_shape = get_BRep_from_file(args.file_path)
    points = get_points_from_BRep(out_shape)
    # print("cooradinate:", points)
    # print("number of vertices:", len(points))
    random.seed(42)
    save_path = args.file_path.replace(".json", ".png").replace("_json", "_img")
    # show_BRep(out_shape=out_shape, show_type=args.show_type, save_path=save_path)
    save_BRep_img(out_shape, save_path)


class ARGS:
    def __init__(self):
        # self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0000/00000007.json"
        self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0002/00020718.json"
        self.save = False
        self.show_type = "body"  # wireframe | body


if __name__ == '__main__':
    args = ARGS()
    run(args)


