import numpy as np
from PIL import Image

def cal_w_ratio(image):
    """
    计算图像中非纯白色像素和纯白色像素的比值。
    支持输入 PIL.Image 或 numpy.ndarray。
    
    Args:
        image: PIL.Image 或 numpy.ndarray (H, W, 3 或 H, W)
    
    Returns:
        float: 非白色像素数量 / 白色像素数量
               如果没有白色像素，返回 float('inf')
               如果没有非白色像素，返回 0.0
               如果图像为空，返回 float('nan')
    """
    # 转换为 numpy
    if isinstance(image, np.ndarray):
        img_array = image
        if img_array.ndim == 2:  # 灰度转三通道
            img_array = np.stack([img_array]*3, axis=-1)
    else:  # PIL
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)

    if img_array.size == 0:
        return float('nan')
    
    # 纯白掩码
    white_mask = (img_array[:, :, 0] == 255) & \
                 (img_array[:, :, 1] == 255) & \
                 (img_array[:, :, 2] == 255)
    
    white_pixels = np.sum(white_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    nonwhite_pixels = total_pixels - white_pixels
    
    if white_pixels == 0:
        if nonwhite_pixels == 0:
            return float('nan')
        else:
            return float('inf')
    else:
        return nonwhite_pixels / white_pixels


def check_border_w(image):
    """
    检测图像最外圈是否全为纯白色。
    支持输入 PIL.Image 或 numpy.ndarray。
    
    Args:
        image: PIL.Image 或 numpy.ndarray (H, W, 3 或 H, W)
    
    Returns:
        bool: True 如果边框全为白色，否则 False
    """
    # 转换为 numpy
    if isinstance(image, np.ndarray):
        img_array = image
        if img_array.ndim == 2:  # 灰度转三通道
            img_array = np.stack([img_array]*3, axis=-1)
    else:  # PIL
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)

    return _check_border_pixels(img_array)


def _check_border_pixels(img_array):
    """
    检查 numpy 图像边框是否全为纯白色
    
    Args:
        img_array: numpy.ndarray, 形状为 (H, W, 3)
    
    Returns:
        bool: 边框是否全白
    """
    height, width = img_array.shape[:2]
    white_color = np.array([255, 255, 255])
    
    if height <= 2 or width <= 2:
        white_mask = (img_array[:, :, 0] == 255) & \
                     (img_array[:, :, 1] == 255) & \
                     (img_array[:, :, 2] == 255)
        return np.all(white_mask)

    # 上边
    if not np.all(img_array[0, :, :] == white_color):
        return False
    # 下边
    if not np.all(img_array[height-1, :, :] == white_color):
        return False
    # 左边
    if not np.all(img_array[1:height-1, 0, :] == white_color):
        return False
    # 右边
    if not np.all(img_array[1:height-1, width-1, :] == white_color):
        return False

    return True


if __name__ == '__main__':
    img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000257/00000257_0.png'
    #img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000006/00000006_0.png'
    img_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary/00000168/00000168_0.png'
    img = Image.open(img_path)
