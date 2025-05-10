import json
import os
import random
import time
import numpy as np
from PIL import Image  # 假设使用Pillow库处理图像
import matplotlib.pyplot as plt  # 可选，用于图像渲染
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe
from cadlib.math_utils import weighted_random_sample
from vis.show_single import save_BRep_img, save_BRep_wire_img, show_BRep
from vis.vis_utils import clip_mask

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox



class ImageProcessor:
    def __init__(self, file_path, output_dir, edit_type='add', edit_path=None, shape_name=None):
        """
        初始化图像处理器
        
        参数:
            config_path (str): JSON配置文件的路径
            output_dir (str): 输出目录，默认为'output'
        """
        self.file_path = file_path
        if shape_name is None:
            self.shape_name = self.file_path.split("/")[-1].split(".")[0] + '.png'
        else:
            self.shape_name = shape_name
        self.output_dir = output_dir
        self.edit_path = edit_path
        self.edit_shape_class = 'mask_box'
        self.seed = int(time.time())

        subdirs = ["init_img", "stroke_img", "mask_img", "process_img", "result_img"]
        for subdir in subdirs:
            save_dir = os.path.join(self.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, self.shape_name)
            setattr(self, subdir+"_output_path", save_path)  # self.init_img_output_path ...

        # 读取形状
        self.init_shape = get_BRep_from_file(self.file_path)
        self.edit_shape = self.get_edit_BRep()
        self.edit_shape_wire = get_wireframe(self.edit_shape)
        # 处理后形状
        self.processed_shape = self.precess_shape(edit_type)

    
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
        save_BRep_img(self.init_shape, self.init_img_output_path, seed=self.seed)
        save_BRep_wire_img(self.edit_shape_wire, self.stroke_img_output_path, seed=self.seed)
        clip_mask(self.stroke_img_output_path, self.mask_img_output_path)
        # 保存编辑后CAD渲染图像
        save_BRep_img(self.processed_shape, self.result_img_output_path, seed=self.seed)
    


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


class ARGS:
    def __init__(self):
        # self.edit_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0000/00000007.json"
        self.edit_path = None
        self.file_path = r"/home/lkh/siga/dataset/deepcad/data/cad_json/0002/00020718.json"
        self.output_dir = r"/home/lkh/siga/dataset/deepcad/data/cad_controlnet_mask"
        self.edit_tpye = 'add' 
        


if __name__ == "__main__":
    args = ARGS()
#{'009923.png', '008733.png', '006746.png'}
    # os.mkdir(args.output_dir)

    for i in range(0,6001):
        shape_name = f"{i:06d}.png"
        processor = ImageProcessor(args.file_path, 
                                    args.output_dir, 
                                    edit_type=args.edit_tpye, 
                                    edit_path=args.edit_path, 
                                    shape_name=shape_name
                                    )
        processor.save_images()

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


