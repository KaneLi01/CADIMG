import json
import h5py
import numpy as np
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepCheck import BRepCheck_Analyzer
import os
import sys

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from vis.show_single import show_BRep
from cadlib.Brep_utils import get_BRep_from_file
from abc import ABC, abstractmethod

# 1

class DatasetMaker(ABC):
    """
    抽象类，用于定义制作数据集的功能
    """

    @abstractmethod
    def generate_initial_image(self, input_data):
        """
        生成初始图像
        :param input_data: 输入数据（例如 CAD 文件路径或其他数据）
        :return: 初始图像
        """
        pass

    @abstractmethod
    def process_image(self, image):
        """
        处理图像（例如生成涂鸦图像或蒙版图像）
        :param image: 输入图像
        :return: 处理后的图像
        """
        pass

    @abstractmethod
    def save_processed_image(self, image, output_path):
        """
        保存处理后的图像
        :param image: 处理后的图像
        :param output_path: 保存路径
        """
        pass


class SimpleDatasetMaker(DatasetMaker):
    """
    从路径 A 读取文件，生成初始图像并保存到路径 B
    """

    def generate_initial_image(self, input_data):
        """
        从输入路径读取文件并生成初始图像
        :param input_data: 输入文件路径
        :return: 初始图像（PIL.Image 对象）
        """
        print(f"Reading file from {input_data}")
        # 示例：假设 input_data 是一个 JSON 文件路径，生成一个空白图像作为初始图像
        initial_image = Image.new("RGB", (512, 512), "white")  # 生成空白图像
        return initial_image

    def process_image(self, image):
        """
        处理图像（此处可以添加自定义处理逻辑）
        :param image: 输入图像
        :return: 处理后的图像
        """
        print("Processing image...")
        # 示例：简单地返回原始图像（无处理）
        return image

    def save_processed_image(self, image, output_path):
        """
        保存处理后的图像到指定路径
        :param image: 处理后的图像
        :param output_path: 保存路径
        """
        print(f"Saving image to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)


def save_BRep_view(file_path, output_dir, display=None):
    # 如果传入display则使用已有的，否则创建新的
    if display is None:
        display, start_display, _, _ = init_display()
        close_display = True
    else:
        close_display = False

    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)
    except Exception as e:
        print("load and create failed.", e)
        return

    display.DisplayShape(out_shape, update=True)
    display.View_Iso()

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{file_name}.png")
    display.View.Dump(output_path)
    print(f"Image saved to {output_path}")

    display.EraseAll()  # 清除显示内容
    if close_display:
        display.Context.Delete()  # 完全关闭显示上下文


def save_single(input_dir, output_dir, filter_invalid=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 只初始化一次显示窗口
    display, start_display, _, _ = init_display()
    
    try:
        files = os.listdir(input_dir)
        files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]  # 只保留文件
        file_path = os.path.join(input_dir, files[0])  
        save_BRep_view(file_path, output_dir, display)
    finally:
        display.Context.Delete()  # 最终关闭显示窗口


def save_single1(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 只初始化一次显示窗口
    display, start_display, _, _ = init_display()
    
    try:
        files = os.listdir(input_dir)
        files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]  # 只保留文件
        file_path = os.path.join(input_dir, files[0])  
        save_path = os.path.join(output_dir, files[0].replace(".json", ".png"))
        out_shape = get_BRep_from_file(file_path)
        show_BRep(out_shape, show_type='body', display=display, save_path=save_path)
    finally:
        display.Context.Delete()  # 最终关闭显示窗口


def save_every_class(input_dir, output_dir):
    # 检查每个类别的图像

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    display, start_display, _, _ = init_display()
    
    try:
        folders = os.listdir(input_dir)
        for folder in folders:
            subfolder = os.path.join(input_dir, folder)
            files = os.listdir(subfolder)
            for i in range(3):
                file_path = os.path.join(subfolder, files[i])
                print(f"Processing file: {file_path}")
                save_BRep_view(file_path, output_dir, display)
            
    finally:
        display.Context.Delete()  # 最终关闭显示窗口


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 只初始化一次显示窗口
    display, start_display, _, _ = init_display()
    
    try:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".json"):
                    print(f"Processing file: {file}")
                    file_path = os.path.join(root, file)
                    save_BRep_view(file_path, output_dir, display)
    finally:
        display.Context.Delete()  # 最终关闭显示窗口



class ARGS:
    def __init__(self):
        self.input_dir = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0005"
        self.output_dir = r"/data/lkunh/datasets/DeepCAD/data/cad_img/0005"
        self.filter = None


if __name__ == '__main__':
    args = ARGS()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # process_directory(args.input_dir, args.output_dir)

    save_single1(args.input_dir, args.output_dir)
    # save_every_class("/data/lkunh/datasets/DeepCAD/data/cad_json", "/data/lkunh/datasets/DeepCAD/data/cad_img/every")


