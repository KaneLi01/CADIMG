import os, ast

from data_cleaning.img_filter import ImageFilter
from data_cleaning.condition_funcs import ImgCheck
from utils.merge_imgs import process_imgs
import utils.path_file_utils as path_file_utils
from src.c1_path import IMGS_DATA_DIR, IMGS_NORMAL_DIR


def filter_invalid_name(imgs_dict, ffunc, whether_filter=True):
    '''过滤无效的name'''
    def delete_first_level_files(directory):
        """
        删除第一级文件
        """
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        os.remove(entry.path)
                    except Exception as e:
                        print(f"删除失败 {entry.name}: {e}")

    for name, dir in imgs_dict.items():
        print(f'==========processing {name}===========')
        output_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check'
        record_txt_path = os.path.join(output_dir, 'record', f'{name}_{ffunc.__name__}.txt')
        copy_target_dir = os.path.join(output_dir, name, ffunc.__name__)
        os.makedirs(os.path.join(output_dir, 'record'), exist_ok=True)
        os.makedirs(copy_target_dir, exist_ok=True)
        imgf = ImageFilter(source_root=dir, record_txt_path=record_txt_path, depth=2, vis_output_dir=copy_target_dir, whether_filter=whether_filter)
    
        imgf.filter_name_vis(ffunc)

        process_imgs(copy_target_dir, os.path.join(copy_target_dir, '00000000'), with_title=True, grid_size=(9, 36))
        delete_first_level_files(copy_target_dir)


def filter_singlecolor_border():
    '''过滤纯色、超出范围'''
    imgs_dict = IMGS_DATA_DIR
    ic = ImgCheck()
    filter_func = ic.check_border  # is_single_color | check_border
    filter_invalid_name(imgs_dict=imgs_dict, ffunc=filter_func)


def filter_normals():
    '''过滤像素过小的normal图'''
    imgs_dict = IMGS_NORMAL_DIR
    ic = ImgCheck()
    filter_func = ic.is_single_color  # is_small_shape | is_single_color
    filter_invalid_name(imgs_dict=imgs_dict, ffunc=filter_func)


def record_sketch():
    '''将sketch图的信息记录在txt文件中'''
    imgs_dict = {"sketch":'/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/sketch/align'}  # /home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/sketch/align | /home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/sketch
    ic = ImgCheck()
    filter_func = ic.count_connected  # count_connected | is_narrow
    filter_invalid_name(imgs_dict=imgs_dict, ffunc=filter_func, whether_filter=False)


def filter_sketch_dash():
    '''通过记录的文件去掉虚线过多的情况'''
    '''
    '11~20': 133351, '21~30': 110989, '1~10': 75935, '41~50': 27596, '31~40': 57331, 
    '71~80': 2327, '61~70': 4895, '51~60': 11963, '81~90': 1298, '121~130': 226, 
    '131~140': 122, '111~120': 281, '91~100': 688, '101~110': 408, '141~150': 45, 
    '151~160': 41, '171~180': 5, '181~190': 2, '161~170': 17'''
    def count_by_intervals(numbers, interval_size=10):
        from collections import Counter
        """统计数字在不同区间的个数"""
        counter = Counter()
        
        for num in numbers:
            # 计算所属区间
            lower = (num - 1) // interval_size * interval_size + 1
            upper = lower + interval_size - 1
            interval_key = f"{lower}~{upper}"
            counter[interval_key] += 1
        
        return dict(counter)
    
    txt_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/sketch_count_connected_aligned.txt'
    data = path_file_utils.read_txt_lines(txt_path)

    for d in data:
        name, num = d.split(',')[0], int(d.split(',')[1])
        if num > 40:
            path_file_utils.append_line_to_file('/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/sketch_dash_filtered.txt', d[:14])



def filter_sketch_narrow():
    '''通过记录的文件去掉狭长的sketch'''
    txt_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/sketch_is_narrow_aligned.txt'
    data = path_file_utils.read_txt_lines(txt_path)

    for d in data:
        if str.isdigit(d[-2]):
            path_file_utils.append_line_to_file('/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/sketch_is_narrow_filtered.txt', d[:14])
    

def main():
    # filter_singlecolor_border()
    # filter_normals()
    # record_sketch()
    filter_sketch_dash()
    # filter_sketch_narrow()


if __name__ == '__main__':
    main()