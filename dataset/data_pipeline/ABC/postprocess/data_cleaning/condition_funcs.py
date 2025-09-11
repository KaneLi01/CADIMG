import numpy as np
import cv2
from PIL import Image, ImageChops

class ImgCheck():
    def check_border(self, img: Image.Image) -> bool:
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


    def is_solid_color(self, img: Image.Image) -> bool:
        """检查图片是否为纯色（所有像素与第一个像素相同）"""
        img_np = np.array(img)
        if img_np.size == 0:
            return True  

        first_pixel = img_np[0, 0] if img_np.ndim == 3 else img_np[0, 0]
        
        return np.all(img_np == first_pixel)


    def is_small_shape(self, img: Image.Image, thre: int = 1600) -> bool:
        """检查图片像素大小"""
        img_np = np.array(img)

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        white = np.array([255, 255, 255])
        
        # 非白色像素
        non_white_pixels = np.any(img_np != white, axis=-1)
        foreground_pixel_count = np.sum(non_white_pixels)    # 像素总数
        
        return foreground_pixel_count < thre


    def is_single_color(self, img: Image.Image) -> bool:
        """检查shape的normal rgb是否小于等于2颜色"""
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


    def is_sketch_closed(self, img: Image.Image) -> bool:
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
            if len(contour) >= 3:
                dist = np.linalg.norm(contour[0][0] - contour[-1][0])
                if dist > 5.0:  # 容差，比如1像素
                    return True  # 有不闭合的轮廓
        return False  # 所有轮廓都闭合


    def count_connected(self, img: Image.Image, thre=64) -> int:
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



    def is_narrow(self, 
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