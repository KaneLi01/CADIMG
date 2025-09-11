from pathlib import Path
from PIL import Image, ImageChops
import os
import utils.img_utils as img_utils
import utils.path_file_utils as path_file_utils
from utils.rootdir_processer import FileProcessor
import numpy as np

class ImageProcessor(FileProcessor):
    def __init__(self, root_dir: str, output_root_dir: str = None, record_txt_path: str = None, aux_root: str = None, depth=2):
        super().__init__(root_dir=root_dir, output_root_dir=output_root_dir, extension='.png', depth=depth)
        self.source_root = Path(root_dir)
        self.output_root_dir = Path(output_root_dir) if output_root_dir else root_dir + "processed"
        self.aux_root = Path(aux_root) if aux_root else None
        self.record_txt_path = Path(record_txt_path) if record_txt_path else None
        os.makedirs(self.output_root_dir, exist_ok=True)

        # 注册函数：单图处理 & 双图处理
        self.single_methods = {
            "flip_b": self.flip_b,
            "extract_contour": self.extract_contour_from_normal,
            "extract_solid_line": self.extract_solid_line,
            "expend": self.expend_pixels,
            "extract_nor_sketch": self.extract_nor_sketch
        }

        self.dual_methods = {
            "stack": self.base_op_stack,
            "align": self.align_contour,
        }

    def process_single(self, method_name: str):
        if method_name not in self.single_methods:
            raise ValueError(f"未知单图方法: {method_name}")
        process_fn = self.single_methods[method_name]

        for img_path in self.iter_files():
            save_path = self._get_output_path(img_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                img = Image.open(img_path).convert("RGB")
                result = process_fn(img)
                if result[0]:
                    path_file_utils.append_line_to_file(self.record_txt_path, os.path.basename(img_path))
                result.save(save_path)

            except Exception as e:
                print(f"[{method_name}] 处理失败 {img_path}: {repr(e)}")


    def process_dual(self, method_name: str):
        if not self.aux_root:
            raise ValueError("未提供辅助图像根目录 aux_root")
        if method_name not in self.dual_methods:
            raise ValueError(f"未知双图方法: {method_name}")
        process_fn = self.dual_methods[method_name]

        for img_path in self.iter_files():
            rel_path = os.path.relpath(img_path, self.root_dir)
            aux_path = os.path.join(self.aux_root, rel_path) 
            save_path = self._get_output_path(img_path)     
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            try:
                img1 = Image.open(img_path).convert("RGB")
                img2 = Image.open(aux_path).convert("RGB")
                result = process_fn(img2, img1)
                result.save(save_path)
            except Exception as e:
                print(f"[{method_name}] 处理失败 {save_path} : {e}")
    

    # === 单图处理 ===

    def flip_b(self, image1: Image.Image) -> Image.Image:
        '''将因为法向量相反导致RGB出错的像素进行矫正'''
        return img_utils.flip_b(image1)
    
    def extract_contour_from_normal(self, image1: Image.Image) -> Image.Image:
        '''提取最外层轮廓'''
        return img_utils.get_contour_img(image1)
    
    def extract_solid_line(self, image1: Image.Image) -> Image.Image:
        '''提取实线轮廓'''
        img_np = np.array(image1)
        result = np.ones_like(img_np) * 255
        not_black = np.any(img_np != [0, 0, 0], axis=-1) 
        not_white = np.any(img_np != [255, 255, 255], axis=-1)  
        target_pixels = not_black & not_white  # 既非黑也非白

        result[target_pixels] = [0, 0, 0]

        return Image.fromarray(result)

    def expend_pixels(self, image1: Image.Image) -> Image.Image:
        return img_utils.expand_pixels(image1)
    
    def extract_nor_sketch(self, image1: Image.Image) -> Image.Image:
        return img_utils.extract_boundaries(image1)

    # === 双图处理 ===

    def base_op_stack(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """
        将op图片叠加在base上，用于处理多实体shape的法向量不一致问题
        """
        return img_utils.stack_imgs(image1, image2, mode='eb')
    
    def align_contour(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        return img_utils.align_by_bbox(image1, image2)

