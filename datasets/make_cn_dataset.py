import json
import os
import random
import time
import sys
sys.path.append("..")

import numpy as np
import cv2
from PIL import Image  # 假设使用Pillow库处理图像
import matplotlib.pyplot as plt  # 可选，用于图像渲染
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe, create_AABB_box, select_BRep_from_file
from cadlib.math_utils import weighted_random_sample
from vis.show_single import save_BRep_img_w_allshape, save_BRep_img_random, save_BRep_wire_img, show_BRep
from vis.vis_utils import clip_mask

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop


class SelectCAD:
    def __init__(self, file_dir, output_dir, shape_name, body_num=2):
        self.file_dir = file_dir
        self.shape_name = shape_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.read_path = os.path.join(self.file_dir, self.shape_name+'.json')
        self.shapes, self.view = select_BRep_from_file(self.read_path, body_num)

        # 判断读取的形状是否有效
        if self.shapes:
            bbox, self.center, ls, self.corners = create_AABB_box(self.shapes['result'])
            if bbox:
                longest_two = sorted(ls)[1:]
                # 舍弃棍子形状的shape
                if longest_two[0] / longest_two[1] > 4 or longest_two[1] / longest_two[0] > 4:
                    self.is_valid = False
                else:

                    # props = GProp_GProps()
                    # brepgprop.VolumeProperties(bbox, props)
                    S_approximate = longest_two[0] * longest_two[1]
                        
                    self.scale = 300 + ((0.75**2) / (round(S_approximate,3)+0.001)) * 200
                    # self.scale = 300 + ((0.75**2) / (round(props.Mass(),3)**(2/3))) * 200

                    
                    self.subdirs = ["base", "operate", "result", "merge"]
                    for subdir in self.subdirs:
                        save_dir = os.path.join(self.output_dir, subdir)
                        os.makedirs(save_dir, exist_ok=True)
                        setattr(self, subdir+"_output_dir", save_dir)  # self.base_output_dir ...
                    self.is_valid = True


        else: self.is_valid = False


    def save_images(self):
        imgs_path = []
        for shape_type in self.subdirs[:-1]:
            output_dir = getattr(self, shape_type+"_output_dir")
            output_name_dir = os.path.join(output_dir, self.shape_name)
            os.makedirs(output_name_dir, exist_ok=True)
            save_BRep_img_w_allshape(self.shapes[shape_type], output_name_dir, scale=self.scale, center=self.center, view=self.view)  
            imgs_path.append(output_name_dir)
        merge_dir = os.path.join(self.merge_output_dir, self.shape_name)
        os.makedirs(merge_dir, exist_ok=True)
        for k in range(0,4):
                
            imgs = [Image.open(os.path.join(img, f'{k}.png')) for img in imgs_path]
            merged_h = np.hstack(imgs)
            cv2.imwrite(os.path.join(merge_dir, f'{k}.png'), merged_h)




class ImageProcessor:
    def __init__(self, file_dir, output_dir, shape_name, edit_type='add', edit_path=None, create_simple_dataset=False):
        """

        """
        self.file_dir = file_dir
        self.shape_name = shape_name
        self.output_dir = output_dir
        self.edit_path = edit_path
        self.edit_shape_class = 'mask_box'
        self.seed = int(time.time())
        self.create_simple_dataset = create_simple_dataset  # 如果只创建简单数据集

        subdirs = ["init_img", "stroke_img", "mask_img", "process_img", "result_img"]
        for subdir in subdirs:
            save_dir = os.path.join(self.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.shape_name+'.png')
            setattr(self, subdir+"_output_path", save_path)  # self.init_img_output_path ...

        # 读取形状
        if self.create_simple_dataset:
            self.init_shape = get_BRep_from_file("/home/lkh/siga/dataset/deepcad/data/cad_json/0002/00020718.json")
        else: 
            self.init_shape = get_BRep_from_file(os.path.join(self.file_dir, self.shape_name+'.json'))
        
        if self.init_shape is None:
            self.is_valid = False
        else:
            self.is_valid = True

        # self.edit_shape = self.get_edit_BRep()
        # self.edit_shape_wire = get_wireframe(self.edit_shape)
        # 处理后形状
        # self.processed_shape = self.precess_shape(edit_type)

    
    def get_edit_BRep(self):
        if self.edit_path is not None:
            return get_BRep_from_file(self.edit_path)
        else:
            return self.generate_BRep(self.edit_shape_class)
        
    def precess_shape(self, edit_type):
        if edit_type == 'add':
            processed_shape = BRepAlgoAPI_Fuse(self.init_shape, self.edit_shape).Shape()
            return processed_shape
        else: raise ValueError("Unsupported edit type. Supported types: 'add'.")

    def generate_BRep(self, shape_class):
        if shape_class == 'box':
            return self.generate_box()
        if shape_class == 'mask_box':
            return self.generate_mask_box()
        else: raise ValueError("Unsupported shape class. Supported classes: 'box'.")
    
    def generate_box(self):
        l, w, h = weighted_random_sample(0.25, 0.75), weighted_random_sample(0.25, 0.75), weighted_random_sample(0.25, 0.75)

        selected_var = random.choice(['x', 'y', 'z'])
        if selected_var == 'x':
            p_x = 0.75
            p_y = weighted_random_sample(0.0, 0.75-w)
            p_z = weighted_random_sample(0.0, 0.75-h)
        elif selected_var == 'y':
            p_y = 0.75
            p_x = weighted_random_sample(0.0, 0.75-l)
            p_z = weighted_random_sample(0.0, 0.75-h)
        else:
            p_z = 0.75
            p_x = weighted_random_sample(0.0, 0.75-l)
            p_y = weighted_random_sample(0.0, 0.75-w)

        # 创建box
        corner = gp_Pnt(p_x, p_y, p_z)
        box = BRepPrimAPI_MakeBox(corner, l, w, h).Shape()
        return box
    
    def generate_mask_box(self):
        l, w, h = weighted_random_sample(0.25, 0.75), weighted_random_sample(0.25, 0.75), weighted_random_sample(0.25, 0.75)
        selected_var = random.choice(['x', 'y', 'z'])
        if selected_var == 'x':
            p_x = 0.0
            selected_var = random.choice(['y', 'z'])
            if selected_var == 'y':
                p_y = 0.75
                p_z = weighted_random_sample(h, 0.75)
            if selected_var == 'z':
                p_z = 0.75
                p_y = weighted_random_sample(w, 0.75)
        elif selected_var == 'y':
            p_y = 0.0
            selected_var = random.choice(['x', 'z'])
            if selected_var == 'x':
                p_x = 0.75
                p_z = weighted_random_sample(h, 0.75)
            if selected_var == 'z':
                p_z = 0.75
                p_x = weighted_random_sample(l, 0.75)
        else:
            p_z = 0.0
            selected_var = random.choice(['x', 'y'])
            if selected_var == 'x':
                p_x = 0.75
                p_y = weighted_random_sample(w, 0.75)
            if selected_var == 'y':
                p_y = 0.75
                p_x = weighted_random_sample(l, 0.75)
        h = -h
        l = -l
        w = -w

        # 创建box
        corner = gp_Pnt(p_x, p_y, p_z)
        box = BRepPrimAPI_MakeBox(corner, l, w, h).Shape()
        return box

    
    def save_images(self):
        # 保存初始CAD图像
        save_BRep_img_random(self.init_shape, self.init_img_output_path, seed=self.seed)
        save_BRep_wire_img(self.edit_shape_wire, self.stroke_img_output_path, seed=self.seed)
        clip_mask(self.stroke_img_output_path, self.mask_img_output_path)
        # 保存编辑后CAD渲染图像
        save_BRep_img_w_allshape(self.processed_shape, self.result_img_output_path, seed=self.seed)
    


def process_image_mask():
    dir = "/data/lkunh/datasets/cad_controlnet02/process_img"
    os.makedirs(dir, exist_ok=True)

    img_dir = "/data/lkunh/datasets/cad_controlnet02/init_img"
    mask_dir = "/data/lkunh/datasets/cad_controlnet02/mask_img"
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    

    assert img_files == mask_files, "img 和 mask 文件名不一致！"

    for img_name, mask_name in zip(img_files, mask_files):
    # 加载 img 和 mask
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = Image.open(img_path).convert("RGB")  # 确保 img 是 RGB 图像
        mask = Image.open(mask_path).convert("L")  # 确保 mask 是灰度图

        # 转换为 NumPy 数组
        img_array = np.array(img, dtype=np.float32)  # [H, W, 3]
        mask_array = np.array(mask, dtype=np.float32)  # [H, W]

        # 将 mask > 254 的部分对应的 img 转换为 -255
        img_array[mask_array > 178] = 0

        # 转换回 PIL 图像并保存
        processed_img = Image.fromarray(img_array.astype(np.uint8))  # 将值限制在 [0, 255]
        processed_img.save(os.path.join(dir, img_name))


def check_loss():
    # 获取目录中所有文件的名称
    dir = r"/data/lkunh/datasets/cad_controlnet02/mask_img"
    existing_files = set(f for f in os.listdir(dir))
    
    # 生成完整的文件名列表
    expected_files = {f"{i:06d}.png" for i in range(0, 9000 + 1)}
    
    # 找到缺失的文件
    missing_files = sorted(int(f.split('.')[0]) for f in (expected_files - existing_files))
    
    print(len(missing_files))


def create_box_dataset(args):
    for i in range(0,6001):
        shape_name = f"{i:06d}.png"
        processor = ImageProcessor(args.file_dir, 
                                    args.output_dir, 
                                    shape_name=shape_name,
                                    edit_type=args.edit_type, 
                                    edit_path=args.edit_path,
                                    create_simple_dataset=True                                
                                    )
         
        # processor.save_images()



def create_init_img_dataset(args):
    i = 0
    dir_path = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    all_dirs = os.listdir(dir_path)  # 0002
    for dir in all_dirs:
        file_dirs = os.path.join(dir_path, dir)  # /.../.../0002
        all_files = os.listdir(file_dirs)  # 00020001.json
        for file in all_files:
            try:
                file_name = file.split('.')[0]  # 00020001
                sc = SelectCAD(file_dirs, args.output_dir, file_name, body_num=2)
                if sc.is_valid:
                    sc.save_images()
                    i += 1
            except Exception as e:
                print(f"{file} failed, pass: {e}")
                continue
        print(f'{dir}中满足要求的形状有：',i)
    print(i)

def test_data():
    # 00020001 2 ；00020002 1 ；00020035 4 ；00020047 10 ； 00005196 分开的body； 00000123 内嵌贴合的body ; 00000544 简单的长方体拼接
    name = '00020346'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    output_dir = "/home/lkh/siga/CADIMG/test"

    sc = SelectCAD(file_dir, output_dir, name, body_num=2)
    if sc.is_valid:
        sc.save_images()



class ARGS:
    def __init__(self):
        # self.file_path = r"/home/lkh/siga/dataset/deepcad/data/cad_json/0002/00020718.json"  # 边长为0.75的正方体
        # self.edit_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0000/00000007.json"
        self.edit_path = None
        self.file_dir = r"/home/lkh/siga/dataset/deepcad/data/cad_json/0002/"
        self.output_dir = r"/home/lkh/siga/dataset/deepcad/data/cad_img/body2_3"
        self.edit_type = 'add' 
        


if __name__ == "__main__":
    args = ARGS()

    # test_data()
    create_init_img_dataset(args)




    # os.mkdir(args.output_dir)


    # shape_name = f"test.png"
    # processor = ImageProcessor(args.file_path, 
    #                             args.output_dir, 
    #                             edit_type=args.edit_tpye, 
    #                             edit_path=args.edit_path, 
    #                             shape_name=shape_name
    #                             )
    # processor.save_images()

    # check_loss()
    # process_image_mask()

    print('finish')

