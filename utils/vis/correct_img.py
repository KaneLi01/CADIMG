import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
import cv2
import numpy as np

from utils import change_bg_img, scale_crop_img, get_contour_img, stack_imgs, process_files_auto


def change_bg_img_fpath(img_path, output_path):
    """
    将blender渲染的图片背景替换为白色
    """
    img = Image.open(img_path)
    change_bg_img(img, output_path)


def scale_crop_img_fpath(img_path, output_path):
    """
    放大图片并裁剪中心区域
    """
    img = Image.open(img_path)
    scale_crop_img(img, output_path)


def get_contour_img_fpath(img_path, output_path):
    """
    获取图片中对象的最外层轮廓，需要传入背景纯白的图片
    """
    img = Image.open(img_path)
    get_contour_img(img, output_path)


def stack_imgs_fpath(img_path1, img_path2, output_path):
    """
    将sketch图片进行叠加
    """
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    stack_imgs(img1, img2, output_path)


def process_sketch_img(img_path1, img_path2, output_path):
    """
    将blender和pythonocc渲染的sketch进行处理，得到用于训练的图
    img1: pythonocc
    img2: blender
    """ 
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2) 

    img1 = scale_crop_img(img1)
    img2 = change_bg_img(img2)
    img2 = get_contour_img(img2)
    _ = stack_imgs(img1, img2, output_path)


def process_img_dataset(init_root, output_root, op_root=None, mode='bg'):
    mode_map = {
        'bg': change_bg_img_fpath,
        'crop': scale_crop_img_fpath,
        'contour': get_contour_img_fpath,
        'stack': stack_imgs_fpath,
        'total': process_sketch_img,
    }
    assert mode in mode_map, f"不支持的模式：{mode}"
    handler = mode_map[mode]

    process_files_auto(
        input_root=init_root,
        output_root=output_root,
        file_handler=handler,
        op_root=op_root,
        suffix_filter='.png'  # 如果你有统一格式要求
    )
    


def main():
    d1 = '/home/lkh/siga/output/temp/con'
    d2 = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views_init/operate/00000272'
    od = '/home/lkh/siga/output/temp/total'
    m = 'total'
    process_img_dataset(init_root=d1, output_root=od, op_root=d2, mode=m)



if __name__ == '__main__':
    main()
