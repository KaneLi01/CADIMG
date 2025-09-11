from PIL import Image, ImageDraw, ImageFont
import os, json
import cv2
import numpy as np


def merge_imgs(
    img_list, 
    save_path, 
    mode='horizontal', 
    grid_size=None, 
    bg_color='white',
    title_list=None
):
    """
    合并图片并保存到指定路径

    参数：
    - img_list: list[Image.Image]，PIL图片对象列表
    - save_path: str，保存路径，例如 'output.png'
    - mode: str，'horizontal' / 'vertical' / 'grid'
    - grid_size: (rows, cols)，当 mode='grid' 时必填
    - bg_color: str，背景色（默认白色）

    返回：
    - merged PIL.Image 对象
    """


    if not img_list:
        raise ValueError("图片列表为空")

    # 确保所有图片尺寸相同（可以扩展成自动 resize）
    w, h = img_list[0].size

    processed_imgs = img_list

    if title_list:
        text_color = 'red'
        font_size = 40
        font = ImageFont.load_default(size=font_size)
        processed_imgs = []
        for img, title in zip(img_list, title_list):
            processed_img = img_add_text(img, img_list[0].size, title, font, text_color)
            processed_imgs.append(processed_img)

    if mode == 'horizontal':
        merged = Image.new('RGB', (w * len(processed_imgs), h), color=bg_color)
        for i, img in enumerate(processed_imgs):
            merged.paste(img, (i * w, 0))

    elif mode == 'vertical':
        merged = Image.new('RGB', (w, h * len(processed_imgs)), color=bg_color)
        for i, img in enumerate(processed_imgs):
            merged.paste(img, (0, i * h))

    elif mode == 'grid':
        if grid_size is None:
            raise ValueError("grid_size 必须指定（行数, 列数）")
        rows, cols = grid_size
        if len(processed_imgs) > rows * cols:
            raise ValueError("图片数量超过网格容量")

        merged = Image.new('RGB', (w * cols, h * rows), color=bg_color)
        for idx, img in enumerate(processed_imgs):
            row, col = divmod(idx, cols)
            merged.paste(img, (col * w, row * h))

    else:
        raise ValueError("mode 应为 'horizontal', 'vertical', 或 'grid'")

    merged.save(save_path)


def img_add_text(img, size, text, font, color):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = size
    # 计算文字位置为上中
    text_width = draw.textlength(text, font=font)
    text_x = (w - text_width) // 2
    text_y = 10  

    draw.text((text_x, text_y), text, fill=color, font=font)

    return img_copy


def scale_crop_img(img, output_path=None, scale=15/13):
    """
    将图片放大后裁剪
    用于将pythonocc渲染的wireframe对齐到blender渲染的normal上
    """
    original_width, original_height = img.size  # 获取图片的长宽
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)  
    
    scaled_img = img.resize((new_width, new_height), Image.NEAREST)  # 缩放图像
    
    # 计算裁剪区域 (中心512x512)
    left = (new_width - original_width) / 2
    top = (new_height - original_height) / 2
    right = left + original_width
    bottom = top + original_height
    
    cropped_img = scaled_img.crop((left, top, right, bottom))
    if output_path:
        cropped_img.save(output_path)
    return cropped_img
    

def get_contour_img(img, output_path=None):
    """
    获取图片中对象形状的最外层轮廓
    用于从blender渲染的图片中获取轮廓，以和pythonocc渲染的wireframe叠加
    """
    img_cv = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    contour_image_cv = np.ones((height, width), dtype=np.uint8) * 255
    cv2.drawContours(contour_image_cv, contours, -1, (0, 255, 0), 2)

    contour_image = Image.fromarray(contour_image_cv)
    if output_path:
        contour_image.save(output_path)
    return contour_image


def find_nonwhite_edges(image):
    """
    检测图像中非纯白和纯白像素的交界处。
    支持输入 PIL.Image 或 numpy.ndarray。
    
    Args:
        image: PIL.Image对象 或 numpy.ndarray (灰度/彩色)
        
    Returns:
        与输入类型一致：
        - 输入 PIL.Image -> 输出 PIL.Image
        - 输入 numpy.ndarray -> 输出 numpy.ndarray (固定为 512x512x3)
    """
    input_is_numpy = isinstance(image, np.ndarray)

    # 如果是 numpy，先规范化为灰度 np.ndarray
    if input_is_numpy:
        if image.ndim == 3:  # 彩色 -> 灰度
            img_array = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            img_array = image.copy()
    else:
        # 输入是 PIL.Image
        if image.mode != 'L':
            image = image.convert('L')
        img_array = np.array(image)

    # 创建输出数组（默认全白）
    height, width = img_array.shape
    edge_array = np.ones((height, width), dtype=np.uint8) * 255
    
    # 创建二值化图像（True为纯白，False为非纯白）
    is_white = img_array == 255
    
    # 方向（8邻域）
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # 遍历像素
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_is_white = is_white[y, x]
            is_edge = False
            for dy, dx in directions:
                neighbor_is_white = is_white[y + dy, x + dx]
                if current_is_white != neighbor_is_white:
                    is_edge = True
                    break
            if is_edge:
                edge_array[y, x] = 0

    if input_is_numpy:
        # 扩展为 (H, W, 3)
        edge_rgb = np.stack([edge_array]*3, axis=-1)
        # 调整到 (512, 512, 3)
        edge_resized = np.array(Image.fromarray(edge_rgb).resize((512, 512), Image.NEAREST))
        return edge_resized
    else:
        return Image.fromarray(edge_array, mode='L')


def stack_imgs(img1, img2, mode='b', output_path=None):
    """
    将图1叠加在图2上。
    mode = 'b'  : 只叠加图2的黑色部分
    mode = 'eb' : 只叠加图2的非黑色部分
    mode = 'ew' : 只叠加图2的非白色部分

    支持输入 PIL.Image 或 numpy.ndarray。
    返回类型与输入类型保持一致。
    """
    input_is_numpy = isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray)

    # 转换为 numpy 数组
    if not isinstance(img1, np.ndarray):
        img1 = np.array(img1.convert("RGB"))
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2.convert("RGB"))

    if img1.shape != img2.shape:
        raise ValueError("A 和 B 图片大小不同，无法逐像素处理")
    
    if mode == 'b':
        mask = np.all(img2 == [0, 0, 0], axis=-1)  
    elif mode == 'eb':
        mask = np.all(img2 != [0, 0, 0], axis=-1)
    elif mode == 'ew':
        mask = np.all(img2 != [255, 255, 255], axis=-1)
    else:
        raise ValueError(f"未知 mode: {mode}")

    img3 = img1.copy()
    img3[mask] = img2[mask]

    if output_path:
        Image.fromarray(img3).save(output_path)

    # 返回类型和输入一致
    if input_is_numpy:
        return img3
    else:
        return Image.fromarray(img3)


def change_bg_img(img, output_path=None, color='white'):
    """
    替换图片的背景颜色。
    用于将blender渲染的图片的灰色背景改为白色/黑色
    """
    img = np.array(img.convert("RGB"))
    bg_color = img[0, 0].copy()

    result_img = img.copy()
    mask = np.all(result_img == bg_color, axis=2)
    if color == 'white':
        result_img[mask] = [255, 255, 255]
    elif color == 'black':
        result_img[mask] = [0, 0, 0]
    else: raise Exception('wrong color')

    result_img = Image.fromarray(result_img)
    if output_path:
        result_img.save(output_path)
    return result_img


def extract_boundaries(img, output_path=None, threshold=30):
    '''根据RGB值提取几何形状的轮廓，尽可能处理圆柱的离散造成的问题'''
    img = img.convert('RGB')
    width, height = img.size
    pixels = img.load()
    
    # 创建输出图像，初始为白
    boundary_img = Image.new('RGB', (width, height), (255, 255, 255))
    boundary_pixels = boundary_img.load()
    
    for y in range(height):
        for x in range(width):
            current_r, current_g, current_b = pixels[x, y]
            is_boundary = False
            
            # 检查四个方向的邻居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                
                # 确保邻居在图像范围内
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_r, neighbor_g, neighbor_b = pixels[nx, ny]
                    
                    # 计算RGB差值
                    diff_r = abs(current_r - neighbor_r)
                    diff_g = abs(current_g - neighbor_g)
                    diff_b = abs(current_b - neighbor_b)
                    
                    # 如果任一差值超过阈值，则是边界
                    if diff_r >= threshold or diff_g >= threshold or diff_b >= threshold:
                        is_boundary = True
                        break
            
            # 如果是边界，设置为红色(或其他明显颜色)
            if is_boundary:
                boundary_pixels[x, y] = (0, 0, 0)  # 红色不透明
    
    # 保存结果
    if output_path:
        boundary_img.save(output_path)
    return boundary_img


def expand_pixels(img, output_path=None):
    '''用于加粗黑色线条'''
    gray = img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    # 创建一个副本用于修改
    result = arr.copy()

    # 找出黑色像素的坐标（值为0）
    black_coords = np.argwhere(arr == 0)

    for y, x in black_coords:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                result[ny, nx] = 0  # 设为黑色

    # 转回 PIL 图像
    result_img = Image.fromarray(result)

    # 如果指定保存路径则保存
    if output_path:
        result_img.save(output_path)

    return result_img    


def align_by_bbox(img1, img2):
    """将 img2 缩放 + 平移，使shape对齐到 img1 上"""

    def get_foreground_bbox(pil_img, threshold=250):
        """获取非白色区域的最小包围盒"""
        gray = pil_img.convert("L")
        arr = np.array(gray)
        
        # 找出非白色像素的位置（前景）
        mask = arr < threshold
        coords = np.argwhere(mask)

        if coords.size == 0:
            raise ValueError("找不到前景对象")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return x_min, y_min, x_max, y_max  # 注意 PIL 是 (x, y)

    # 获取 bbox
    x1_min, y1_min, x1_max, y1_max = get_foreground_bbox(img1)
    x2_min, y2_min, x2_max, y2_max = get_foreground_bbox(img2)

    # 尺寸
    w1, h1 = x1_max - x1_min, y1_max - y1_min
    w2, h2 = x2_max - x2_min, y2_max - y2_min

    # 缩放比例
    scale_x = w1 / w2
    scale_y = h1 / h2
    scale = min(scale_x, scale_y)  # 保证前景能整体放入

    # 缩放 img2
    img2_resized = img2.resize(
        (int(img2.width * scale), int(img2.height * scale)),
        Image.BICUBIC
    )

    # 重新获取缩放后的 bbox（粗略估计）
    x2_min_scaled = int(x2_min * scale)
    y2_min_scaled = int(y2_min * scale)

    # 偏移量 = 对准 img1 的 bbox 左上角
    offset_x = x1_min - x2_min_scaled
    offset_y = y1_min - y2_min_scaled

    # 创建和 img1 一样大小的画布（白背景）
    aligned_img = Image.new("RGB", img1.size, color="white")
    aligned_img.paste(img2_resized, (offset_x, offset_y))

    return aligned_img


def resize_imgs():
    input_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp1'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp2'

    img_names = os.listdir(input_dir)
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        process_image(img_path, output_path)



    
def save_contour():
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "sketch single circle"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir

    input_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views_temp/sketch_img'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp3'
    for i, name in enumerate(dir['file_names']):
        for i in range(0,6):
            img_path = os.path.join(input_dir, name+f'_{i}.png')
            output_path = os.path.join(output_dir, name+f'_{i}.png')
            get_contour(img_path, output_path)

    # get_contour(img_path, output_path)


def flip_b(image: Image.Image, output_path: str=None):
    '''根据B通道翻转RGB像素，用于处理法向量错误的部分'''
    img_array = np.asarray(image).astype(np.float32) / 255.0

    # B通道
    blue_channel = img_array[:, :, 2]

    mask = blue_channel < 0.5
    

    transformed = 1 - img_array
    transformed = np.clip(transformed, 0.0, 1.0)
    img_array[mask] = transformed[mask]
    result_array = (img_array * 255).astype(np.uint8)

    result_image = Image.fromarray(result_array)

    if output_path:
        result_image.save(output_path)
    if np.any(mask == 1):
        return True, result_image
    else: return False, result_image


def analyze_frequency_complexity(img, high_freq_threshold_ratio=0.3):
    """
    输入一个图像路径，分析其频率域的高频成分占比，作为复杂度指标
    :param image_path: 图片路径
    :param high_freq_threshold_ratio: 高频的“中心排除半径”的比例（越小，越严格）
    :return: 高频能量比例，0~1之间，越高说明越复杂
    """
    img = img.convert('L')
    img = np.array(img)
    # 执行傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 中心位置
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)

    # 计算高频区域
    radius = high_freq_threshold_ratio * max_dist
    high_freq_mask = dist > radius

    # 计算能量总和和高频能量比例
    total_energy = magnitude.sum()
    high_freq_energy = magnitude[high_freq_mask].sum()
    high_freq_ratio = high_freq_energy / total_energy

    return high_freq_ratio


def test():
    # /home/lkh/siga/dataset/ABC/temp/processed_imgs/source/result_oppo/00/00000126/00000126_0.png
    # /home/lkh/siga/dataset/ABC/temp/processed_imgs/source/result_oppo/00/00000446/00000446_0.png
    root_dir = '/home/lkh/siga/output/temp/722/'
    base_dir = os.path.join(root_dir, 'base')
    sketch_dir = os.path.join(root_dir, 'sketch0')
    
    name_list = os.listdir(base_dir)
    for name in name_list:
        img_path1 = os.path.join(base_dir, name)
        img_path2 = os.path.join(sketch_dir, name)
        if not os.path.exists(img_path2):
            continue
        output_path = os.path.join(root_dir, 'input', name)
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        r = stack_imgs(img1, img2, mode='eb', output_path=output_path)

    # img_path1 = '/home/lkh/siga/output/temp/722/base/pos_00960324_2.png'
    # img_path2 = '/home/lkh/siga/output/temp/722/sketch0/pos_00960324_2.png'
    # output_path = '/home/lkh/siga/output/temp/722/input/pos_00960324_2.png'
    # img1 = Image.open(img_path1)
    # img2 = Image.open(img_path2)
    # r = stack_imgs(img1, img2, mode='eb', output_path=output_path)


if __name__ == "__main__":
    
    test()
