import numpy as np
from PIL import Image

def cal_bw_ratio(image):
    """
    计算图像中纯黑色像素和纯白色像素的比值
    
    Args:
        image: PIL.Image对象
    
    Returns:
        float: 黑色像素数量 / 白色像素数量的比值
               如果没有白色像素，返回 float('inf')
               如果没有黑色像素，返回 0.0
               如果都没有，返回 float('nan')
    """
    # 确保图像是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 找到纯黑色像素 (R=0, G=0, B=0)
    black_pixels = np.sum((img_array[:, :, 0] == 0) & 
                         (img_array[:, :, 1] == 0) & 
                         (img_array[:, :, 2] == 0))
    
    # 找到纯白色像素 (R=255, G=255, B=255)
    white_pixels = np.sum((img_array[:, :, 0] == 255) & 
                         (img_array[:, :, 1] == 255) & 
                         (img_array[:, :, 2] == 255))
    
    # 计算比值
    if white_pixels == 0 and black_pixels == 0:
        return float('nan')  # 既没有纯黑也没有纯白
    elif white_pixels == 0:
        return float('inf')  # 只有黑色，没有白色
    else:
        return black_pixels / white_pixels
    

def check_border_w(image):
    """
    检测PIL图像最外圈是否全为纯白色
    
    Args:
        image: PIL.Image对象
    
    Returns:
        bool: 如果最外圈全为纯白色返回True，否则返回False
    """
    # 确保图像是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    return _check_border_pixels(img_array)


def _check_border_pixels(img_array):
    """
    检查图像边框像素是否全为纯白色
    
    Args:
        img_array: numpy数组，形状为 (height, width, 3)
    
    Returns:
        bool: 边框全为白色返回True，否则返回False
    """
    height, width = img_array.shape[:2]
    
    # 如果图像太小（宽度或高度小于等于2），直接检查所有像素
    if height <= 2 or width <= 2:
        white_mask = (img_array[:, :, 0] == 255) & \
                     (img_array[:, :, 1] == 255) & \
                     (img_array[:, :, 2] == 255)
        return np.all(white_mask)
    
    # 定义纯白色 (255, 255, 255)
    white_color = np.array([255, 255, 255])
    
    # 检查四条边是否全为白色
    
    # 上边 (第一行)
    top_edge = img_array[0, :, :]
    if not np.all(np.all(top_edge == white_color, axis=1)):
        return False
    
    # 下边 (最后一行)
    bottom_edge = img_array[height-1, :, :]
    if not np.all(np.all(bottom_edge == white_color, axis=1)):
        return False
    
    # 左边 (第一列，排除已检查的角落)
    left_edge = img_array[1:height-1, 0, :]
    if not np.all(np.all(left_edge == white_color, axis=1)):
        return False
    
    # 右边 (最后一列，排除已检查的角落)
    right_edge = img_array[1:height-1, width-1, :]
    if not np.all(np.all(right_edge == white_color, axis=1)):
        return False
    
    return True


if __name__ == '__main__':
    img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000257/00000257_0.png'
    #img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000006/00000006_0.png'
    img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000168/00000168_0.png'
    img = Image.open(img_path)
    print(cal_bw_ratio(img))