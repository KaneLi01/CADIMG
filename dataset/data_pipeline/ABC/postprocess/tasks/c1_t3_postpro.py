import os
import numpy as np
from PIL import Image

from data_cleaning.img_filter import ImageFilter
from data_cleaning.img_processer import ImageProcessor
from data_cleaning.condition_funcs import ImgCheck
from src.c1_path import IMGS_DATA_DIR, IMGS_NORMAL_DIR

output_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check'


def process_RGB():
    '''将图片RGB修正'''
    imgs_dict = IMGS_NORMAL_DIR
    for name, dir in imgs_dict.items():
        print(f'==========processing {name}===========')
        pro_target_dir = os.path.join(output_dir, name, 'flip_RGB')
        record_txt_path = os.path.join(output_dir, 'record', f'{name}_wrongRGB.txt')
        os.makedirs(pro_target_dir, exist_ok=True)
        imgp = ImageProcessor(root_dir=dir, output_root_dir=pro_target_dir, depth=2, record_txt_path=record_txt_path)
        imgp.process_single(method_name='flip_b')


def process_mask():
    '''保存mask'''
    def white_mask(img_path):
        # 打开图像并转为RGB
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

        # 创建布尔掩码：像素是否等于 (255, 255, 255)
        mask = np.all(arr == [255, 255, 255], axis=-1)

        # 转换成 0/1 数组
        binary = mask.astype(np.uint8)

        return binary
    
    input_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/add'

    pro_target_dir = os.path.join(output_dir, 'normal_add', 'mask')
    os.makedirs(pro_target_dir, exist_ok=True)
    imgp = ImageProcessor(root_dir=input_dir, output_root_dir=pro_target_dir, depth=2)
    for img_path in imgp.iter_files():
        save_path = imgp._get_output_path(img_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        m = white_mask(img_path)
        mask_img = Image.fromarray((m * 255).astype(np.uint8))
        mask_img.save(save_path)


def process_align():
    sketch_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/add'

    pro_target_dir = os.path.join(output_dir, 'sketch', 'align')
    os.makedirs(pro_target_dir, exist_ok=True)
    imgp = ImageProcessor(root_dir=sketch_dir, output_root_dir=pro_target_dir, aux_root=normal_dir, depth=2)

    imgp.process_dual(method_name='align')


def main():
    process_mask()
    #process_align()

if __name__ == '__main__':
    main()