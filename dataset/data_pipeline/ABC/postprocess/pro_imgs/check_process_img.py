import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.path_file_utils as file_utils
import utils.img_utils as img_utils
import utils.jsonl_utils as jsonl_utils
from PIL import Image, ImageChops
import shutil
from pathlib import Path
import numpy as np
import cv2


def check_border(img: Image.Image) -> bool:
    '''检查图像的边框，是否图形超出摄像机范围'''
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    edges = [
        img_np[0, :, :],   
        img_np[-1, :, :],  
        img_np[:, 0, :],   
        img_np[:, -1, :]   
    ]
    return any(np.any(edge != 255) for edge in edges)


def is_solid_color(img: Image.Image) -> bool:
    """检查图片是否为纯色（所有像素与第一个像素相同）"""
    img_np = np.array(img)
    if img_np.size == 0:
        return True  

    first_pixel = img_np[0, 0] if img_np.ndim == 3 else img_np[0, 0]
    
    return np.all(img_np == first_pixel)


def is_small_shape(img: Image.Image, thre: int = 1600) -> bool:
    """检查图片像素大小"""
    img_np = np.array(img)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    
    white = np.array([255, 255, 255])
    
    # 非白色像素
    non_white_pixels = np.any(img_np != white, axis=-1)
    foreground_pixel_count = np.sum(non_white_pixels)    # 像素总数
    
    return foreground_pixel_count < thre


def is_single_color(img: Image.Image) -> bool:
    """检查shape的normal rgb是否小于等于2中颜色"""
    img_np = np.array(img)
    
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    white = np.array([255, 255, 255])

    non_white_mask = np.any(img_np != white, axis=-1)  
    non_white_pixels = img_np[non_white_mask]          

    if len(non_white_pixels) == 0:
        return True

    unique_colors = np.unique(non_white_pixels, axis=0) 
    num_unique_colors = len(unique_colors)
    
    return num_unique_colors <= 2


def is_sketch_closed(img: Image.Image) -> bool:
    """
    检测前景轮廓线是否闭合
    
    参数:
        img: PIL.Image 对象（白色背景，前景为轮廓线）
    
    返回:
        bool: True（闭合）或 False（未闭合）
    """
    # 转换为 OpenCV 格式 (灰度图)
    img_np = np.array(img)
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 二值化：非白色=255（前景），白色=0
    _, binary = cv2.threshold(img_np, 254, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 检查轮廓是否闭合（轮廓首尾相接）
        print(contour)
        print('-----')
        if len(contour) >= 3:
            dist = np.linalg.norm(contour[0][0] - contour[-1][0])
            if dist > 5.0:  # 容差，比如1像素
                return False  # 有不闭合的轮廓
    return True  # 所有轮廓都闭合


def count_connected(img: Image.Image) -> int:
    """
    统计图片中被白色包围的独立非白色像素块数量
    
    参数:
        img: PIL.Image 对象（背景为白色）
    
    返回:
        int: 独立像素块的数量
    """
    # 转换为 OpenCV 格式 (BGR)
    img_np = np.array(img)
    if img_np.ndim == 2:  # 灰度图转 RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    # 定义白色 (BGR=255,255,255)
    white = np.array([255, 255, 255])
    
    # 生成二值掩码：非白色=1，白色=0
    mask = np.any(img_np != white, axis=-1).astype(np.uint8) * 255
    
    # 连通区域标记
    num_labels, labels = cv2.connectedComponents(mask)
    
    # 减去背景（背景算作 label=0）
    return num_labels - 1


def is_narrow(
    img: Image.Image,
    output_path: str = "output.png",
    min_side_length: int = 6,        # 最小边长阈值
    aspect_ratio_thresh: float = 16,  # 长宽比阈值（用于面积比判断）
    extreme_aspect_ratio: float = 30, # 极端长宽比阈值（新增）
    area_ratio_thresh: float = 0.125 # 面积比阈值(r2 = 1/8)
) -> None:
    
    # 转换为 OpenCV 格式并二值化
    img_np = np.array(img)
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV) 
    # cv2.imwrite('/home/lkh/siga/output/temp/test/1.png', binary)  # 保存二值化图像用于调试

    # 查找轮廓（找的是白色部分）
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个新白图用于绘制结果
    output_img = np.ones_like(img) * 255

    for i, cnt in enumerate(contours):    
        if hierarchy[0][i][2] == -1:
            # print(i)
            # import random
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # cv2.drawContours(output_img, [cnt], -1, color, 2)

            # 计算最小外接矩形
            rect = cv2.minAreaRect(cnt)
            (_, _), (w, h), _ = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # 计算关键指标
            min_side = min(w, h)
            contour_area = cv2.contourArea(cnt)
            rect_area = w * h
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            area_ratio = contour_area / rect_area if rect_area > 0 else 0
            
            # 判断条件
            if min_side < min_side_length and contour_area > 50:
                # 条件1：存在边长<5的极小包围盒（红色）
                # cv2.drawContours(output_img, cnt, -1, (0, 0, 122), 2)
                # cv2.drawContours(output_img, [box], 0, (0, 0, 255), 2)
                return True, 1
            elif aspect_ratio > extreme_aspect_ratio:
                # 条件2（新增）：长宽比>30的极端狭长形状（蓝色）
                # cv2.drawContours(output_img, [box], 0, (255, 0, 0), 2)
                return True, 2
            elif aspect_ratio / aspect_ratio_thresh > area_ratio and contour_area > 100:
                # 条件3：长宽比/8 > 面积比的普通狭长形状（绿色）
                # drawContours(output_img, [box], 0, (0, 255, 0), 2)
                return True, 3

        # 保存结果
            # cv2.imwrite(output_path+f'{i}.png', output_img)
    # cv2.imwrite(output_path, output_img)
    return False, None


class ImageProcessor:
    def __init__(self, source_root: str, target_root: str = None, aux_root: str = None):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root) if target_root else source_root + "processed"
        self.aux_root = Path(aux_root) if aux_root else None

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

        for src_file, rel_path in self._iter_images():
            dst_file = self.target_root / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                img = Image.open(src_file).convert("RGB")
                result = process_fn(img)
                result.save(dst_file)
                print(f"[{method_name}] 已保存: {dst_file}")
            except Exception as e:
                print(f"[{method_name}] 处理失败 {src_file}: {e}")

    def process_dual(self, method_name: str):
        if not self.aux_root:
            raise ValueError("未提供辅助图像根目录 aux_root")
        if method_name not in self.dual_methods:
            raise ValueError(f"未知双图方法: {method_name}")
        process_fn = self.dual_methods[method_name]

        for src_file, rel_path in self._iter_images():
            aux_file = self.aux_root / rel_path
            dst_file = self.target_root / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
        
            if not aux_file.exists():
                print(f"跳过缺失辅助图像: {aux_file}")
                continue

            try:
                img1 = Image.open(src_file).convert("RGB")
                img2 = Image.open(aux_file).convert("RGB")
                result = process_fn(img1, img2)
                result.save(dst_file)
                print(f"[{method_name}] 已保存: {dst_file}")
            except Exception as e:
                print(f"[{method_name}] 处理失败 {src_file} 和 {aux_file}: {e}")
    
    def filter_name(self, condition_fn: callable, output_txt: str, mode=None):
        '''需要从外部传入条件函数进行筛选'''
        recorded_dirs = set()

        for src_file, rel_path in self._iter_images():
            try:
                img = Image.open(src_file).convert("RGB")
                if mode == 'dot_num':
                    p = str(rel_path).split('/')[-1].split('.')[0]
                    c = condition_fn(img)
                    line = p + ', ' + str(c)
                    recorded_dirs.add(line)
                if mode == 'narrow':
                    p = str(rel_path).split('/')[-1].split('.')[0]
                    b, result = condition_fn(img)
                    if b:
                        line = p + ', ' + str(result)
                        recorded_dirs.add(line)                    
                else:
                    if condition_fn(img):
                        p = str(rel_path).split('/')[-1].split('.')[0]  # 获取相对父目录路径
                        recorded_dirs.add(p)
            except Exception as e:
                print(f"检查失败 {src_file}: {e}")

        # 写入txt文件
        with open(output_txt, "w") as f:
            f.write("\n".join(sorted(recorded_dirs)))
        print(f"已记录 {len(recorded_dirs)} 个目录到 {output_txt}")

    def _iter_images(self):
        '''返回png文件相对于root的路径'''
        for i in range(100):
            subdir = f"{i:02d}"
            subdir_path = self.source_root / subdir
            if not subdir_path.exists():
                continue
            for dirpath, _, filenames in os.walk(subdir_path):
                for filename in filenames:
                    if filename.lower().endswith('.png'):
                        src_file = Path(dirpath) / filename
                        rel_path = src_file.relative_to(self.source_root)
                        yield src_file, rel_path

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


    '''对比operation和target'''
    def compare_and_record_same_images(self, compare_root: str, output_txt: str, tolerance: int = 600):
        """
        比较两个目录下相同相对路径的图片是否相同，并将相同图片的路径记录到txt文件
        
        参数:
            compare_root: 要对比的另一个根目录路径
            output_txt: 结果输出文件路径
            tolerance: 允许的像素差异容差(0表示必须完全相同)
        """
        compare_root = Path(compare_root)
        same_images = set()

        for src_file, rel_path in self._iter_images():
            compare_file = compare_root / rel_path
            
            if not compare_file.exists():
                continue

            try:
                img1 = Image.open(src_file).convert("RGB")
                img2 = Image.open(compare_file).convert("RGB")
                
                if self._images_equal(img1, img2, tolerance):
                    p = str(rel_path).split('/')[-1].split('.')[0]  
                    same_images.add(p)
            except Exception as e:
                print(f"比较失败 {src_file} 和 {compare_file}: {e}")

        # 写入结果文件
        with open(output_txt, "w") as f:
            f.write("\n".join(sorted(same_images)))
        
        print(f"找到 {len(same_images)} 张相同图片，结果已保存到 {output_txt}")

    def _images_equal(self, img1: Image.Image, img2: Image.Image, tolerance: int = 0) -> bool:
        """比较两张图片是否相同(考虑容差)"""
        if img1.size != img2.size or img1.mode != img2.mode:
            return False

        diff = ImageChops.difference(img1, img2)
        diff_pixels = sum(1 for pixel in diff.getdata() if pixel != (0, 0, 0))
        return diff_pixels <= tolerance


def check_img_num():
    '''检查obj和img文件数量是否正确'''
    # 首先检查name数量
    root_dir = {
        'obj': '/home/lkh/siga/dataset/ABC/temp/obj',
        'normal': '/home/lkh/siga/dataset/ABC/temp/normal'
    }
    txt_dir = '/home/lkh/siga/CADIMG/dataset/render_and_postprocess/ABC/src/'
    
    for key, value in root_dir.items():
        txt_path = txt_dir + key + '_file_num2.txt'
        with open(txt_path, "w") as f:
            type_dirs_rela = sorted(os.listdir(value))  # [base
            for type_dir_rela in type_dirs_rela:
                type_dir_abs = os.path.join(value, type_dir_rela)
                cls_dirs_rela = sorted(os.listdir(type_dir_abs))  # [00]
                for cls_dir_rela in cls_dirs_rela:
                    count = 0
                    cls_dir_abs = os.path.join(type_dir_abs, cls_dir_rela)
                    name_dirs_rela = sorted(os.listdir(cls_dir_abs))  # [00000000]
                    for name_dir_rela in name_dirs_rela:
                        name_dir_abs = os.path.join(cls_dir_abs, name_dir_rela)
                        file_paths = os.listdir(name_dir_abs)
                        l = len(file_paths)
                        if key == 'obj':
                            if l == 1:
                                count += 1
                        elif key == 'normal':
                            if l == 8:
                                count += 1
                    line = f"{type_dir_rela}, {cls_dir_rela}, {count}\n"
                    f.write(line)


def delete_name_notin_jsonl(root_dir, mode='check'):
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/ff_feats_sorted.jsonl'
    
    cls_dirs_rela = sorted(os.listdir(root_dir))
    for cls_dir_rela in cls_dirs_rela:
        cls_dir_abs = os.path.join(root_dir, cls_dir_rela)
        cls_dirs_rela = sorted(os.listdir(cls_dir_abs))
        for cls_dir_rela in cls_dirs_rela:
            if not jsonl_utils.find_dic(jsonl_path, 'name', cls_dir_rela):
                if mode == 'check':
                    print(cls_dir_rela)
                elif mode == 'del':
                    delete_dir = os.path.join(cls_dir_abs, cls_dir_rela)
                    shutil.rmtree(delete_dir)


def test():
    # /home/lkh/siga/dataset/ABC/rough_data/pos/sketch/00/00000931/00000931_3.png
    # /home/lkh/siga/dataset/ABC/rough_data/pos/sketch/00/00001283/00001283_1.png
    # /home/lkh/siga/dataset/ABC/temp/processed_imgs/source/soild_trans/00/00000512/00000512_1.png
    img_path = '/home/lkh/siga/dataset/ABC/temp/processed_imgs/source/soild_trans/00/00000038/00000038_1.png'
    img = Image.open(img_path)
    is_narrow(img, output_path="/home/lkh/siga/output/temp/test/result_with_rectangles.png")



def filter_by_different_condition(mode=None):

    if mode == 'blank':
        shape_types = ['base', 'operation', 'sketch', 'target']
        condition_fn = is_solid_color
    elif mode == 'size':
        shape_types = ['base', 'operation']
        condition_fn = is_small_shape
    elif mode == 'single_color':
        shape_types = ['base', 'operation']
        condition_fn = is_single_color
    elif mode == 'dot_num':
        shape_types = ['sketch']
        condition_fn = count_connected
    elif mode == 'closed':
        shape_types = ['sketch']
        condition_fn = is_sketch_closed
    elif mode == 'narrow':
        shape_types = ['solid']
        condition_fn = is_narrow
    
    root_dir = '/home/lkh/siga/dataset/ABC/rough_data'
    txt_root_dir = '/home/lkh/siga/dataset/ABC/rough_data/record'
    rela_dir = ['pos', 'neg']

    for rela in rela_dir:
        parent_dir = os.path.join(root_dir, rela)

        for shape_type in shape_types:
            dir = os.path.join(parent_dir, shape_type)
            txt_dir = os.path.join(txt_root_dir, rela, mode)
            os.makedirs(txt_dir, exist_ok=True)
            txt_path = os.path.join(txt_dir, f"{rela}_{shape_type}_{mode}.txt")

            processor = ImageProcessor(
                source_root=dir
            )   
            
            processor.filter_name(condition_fn=condition_fn, output_txt=txt_path, mode=mode)


def compare_same_op_target():

    root_dir = '/home/lkh/siga/dataset/ABC/rough_data'
    txt_root_dir = '/home/lkh/siga/dataset/ABC/rough_data/record'
    rela_dir = ['pos', 'neg']

    for rela in rela_dir:
        parent_dir = os.path.join(root_dir, rela)

        dir = os.path.join(parent_dir, 'operation')
        com_dir = os.path.join(parent_dir, 'target')
        txt_dir = os.path.join(txt_root_dir, rela, 'cover')
        os.makedirs(txt_dir, exist_ok=True)
        txt_path = os.path.join(txt_dir, f"{rela}_operation_cover.txt")

        processor = ImageProcessor(source_root=dir)
        processor.compare_and_record_same_images(
            compare_root=com_dir,
            output_txt=txt_path,
            tolerance=600  # 可选: 允许的像素差异容差
        )


def generation_sketch(mode=None):

    if mode == '99':
        shape_types = ['render_normal']
        method_name = 'extract_nor_sketch'

    root_dir = '/home/lkh/siga/output/temp/722'

    for shape_type in shape_types:
        dir = os.path.join(root_dir, shape_type)

        processor = ImageProcessor(
            source_root=dir,
            target_root='/home/lkh/siga/output/temp/722/sketch1',
        )   
        
        processor.process_single(method_name=method_name)


if __name__ == '__main__':
    # dir = '/home/lkh/siga/dataset/ABC/temp/normal/sketch_oppo'
    # delete_name_notin_jsonl(dir,mode='check')
    generation_sketch(mode='99')
    # test()
    # mode = 'narrow'  # 'blank', 'size', 'single_color', 'dot_num', 'closed'
    # filter_by_different_condition(mode)