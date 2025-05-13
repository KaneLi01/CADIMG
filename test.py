from utils import log_util
from vis import show_single
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import random

from OCC.Core.gp import gp_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.SimpleGui import init_display

from vis.show_single import show_BRep
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe


def create_2_box():
    p1_1 = gp_Pnt(-1.0, -1.0, -1.0)
    p1_2 = gp_Pnt(1.0, 1.0, 1.0)
    box1 = BRepPrimAPI_MakeBox(p1_1, p1_2)
    shape1 = box1.Shape()

    p2_1 = gp_Pnt(0.0, 0.0, 0.0)
    p2_2 = gp_Pnt(2.0, 2.0, 2.0)    
    box2 = BRepPrimAPI_MakeBox(p2_1, p2_2)
    shape2 = box2.Shape()

    return shape1, shape2


def create_AABB_box(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    lower = gp_Pnt(xmin, ymin, zmin)
    upper = gp_Pnt(xmax, ymax, zmax)  
    print(xmax-xmin, ymax-ymin, zmax-zmin)
    return BRepPrimAPI_MakeBox(lower, upper).Shape()


def test_AABB():
    '''测试并可视化包围盒'''
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)
    aabb_box = create_AABB_box(shape)

    aabb_wires = get_wireframe(aabb_box)

    display, start_display, _, _ = init_display()

    for edge in aabb_wires:
        display.DisplayShape(edge, update=True, color="black")


    display.DisplayShape(shape, update=True)
    start_display()

def test_volume():
    '''测试相交体积'''
    shape1, shape2 = create_2_box()
    body = BRepAlgoAPI_Common(shape1, shape2).Shape()
    props = GProp_GProps()
    brepgprop.VolumeProperties(body, props)
    print(props.Mass())


def test_shape_distance():
    '''测试两个shape之间的最短距离'''
    '''创建两个box作为shape'''
    shape1, shape2 = create_2_box()

    # 计算两个box的最小距离
    dist = BRepExtrema_DistShapeShape(shape1, shape2)
    if dist.IsDone() and dist.Value() < 1e-6:  # 考虑浮点误差
        print(dist.Value())
    else: print(dist.Value())


def main():
    test_AABB()


if __name__ == "__main__":
    main()


# img_path = "/home/lkh/siga/CADIMG/infer"
# imgs = []
# for i in range(0,4):
#     img_p = os.path.join(img_path, f"{i:06d}.png")
#     imgs.append(np.array(Image.open(img_p)))
# images_np = [np.array(img) for img in imgs]
# result_np = np.concatenate(images_np, axis=1)  # axis=1 表示水平

# # 转回 PIL.Image 并保存
# result = Image.fromarray(result_np)
# result.save(os.path.join(img_path, "02.png"))

# img = Image.open(img_path).convert("L")
# img0 = np.array(img)
# print(img0.shape)
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# img = transform(img)
# print(img.shape)
# img = transforms.ToPILImage()(img)
# img1 = np.array(img)
# print(img1.shape)
# transform = transforms.Compose([
#                 transforms.Resize((128,128)),
#                 transforms.ToTensor()
#             ])

# mask = transform(mask)
# print(mask[0,:,64])
