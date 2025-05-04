from PIL import Image
import cv2
import numpy as np
import os

def crop_to_center_512(input_path, output_path=None):
    """
    将输入图片裁剪为中心 512x512 大小的正方形
    
    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径（可选，默认覆盖原文件）
    
    返回:
        bool: 是否成功
    """
    try:
        # 打开图片
        img = Image.open(input_path)
        
        # 获取原始尺寸
        width, height = img.size
        
        # 计算裁剪区域（取中心正方形）
        crop_size = min(width, height)  # 选择短边作为基准
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # 裁剪中心区域
        img_cropped = img.crop((left, top, right, bottom))
        
        # 缩放到 512x512
        img_resized = img_cropped.resize((512, 512), Image.LANCZOS)
        
        # 保存（如果未指定输出路径，覆盖原文件）
        if output_path is None:
            output_path = input_path
        img_resized.save(output_path)
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    

def clip_mask(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # 二值化
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    mask = np.zeros_like(image, dtype=np.uint8)  # 确保 dtype=np.uint8
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # 填充轮廓内部

    # 内部区域（填充部分）
    inner_region = mask
    # 外部区域（背景）
    outer_region = cv2.bitwise_not(mask)

    cv2.imwrite(output_path, inner_region)
    

# if __name__ == "__main__":
#     image_path = "/data/lkunh/datasets/DeepCAD/data/cad_img/0002/00020718.png"
#     crop_to_center_512(image_path)


def process_image_mask(img_path, mask_path, output_path):
    img = Image.open(img_path).convert("RGB")  # 确保 img 是 RGB 图像
    mask = Image.open(mask_path).convert("L")  # 确保 mask 是灰度图

    # 转换为 NumPy 数组
    img_array = np.array(img, dtype=np.float32)  # [H, W, 3]
    mask_array = np.array(mask, dtype=np.float32)  # [H, W]

    # 将 mask > 254 的部分对应的 img 转换为 -255
    img_array[mask_array > 178] = 0

    # 转换回 PIL 图像并保存
    processed_img = Image.fromarray(img_array.astype(np.uint8))  # 将值限制在 [0, 255]
    processed_img.save(output_path)