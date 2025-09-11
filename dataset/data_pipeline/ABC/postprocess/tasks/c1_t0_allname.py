import os, shutil

from utils.rootdir_processer import FileProcessor
import utils.path_file_utils as path_file_utils
from src.c1_path import IMGS_DATASET_DIR


def remove_name():
    txt_root_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record'
    all_path = os.path.join(txt_root_dir, 'all.txt')
    all_name = path_file_utils.read_txt_lines(all_path)
    all_names_set = set(all_name)

    txt_names = [
        'normal_add_check_border', 
        'normal_add_is_single_color',
        'normal_add_is_small_shape',
        'normal_add_wrongRGB',
        'normal_base_is_single_color',
        'normal_base_wrongRGB',
        'normal_target_is_single_color',
        'sketch_dash_filtered',
        'sketch_is_narrow_filtered'
        ]

    all_removed_names = set()
    for txt_name in txt_names:
        remove_txt_path = os.path.join(txt_root_dir, f'{txt_name}.txt')
        removed_names = path_file_utils.read_txt_lines(remove_txt_path)
        all_removed_names.update(removed_names)
    remaining_names = sorted(list(all_names_set - all_removed_names))
    path_file_utils.write_list_to_txt(remaining_names, os.path.join(txt_root_dir, 'filtered.txt'))


def get_all_name():
    root_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/sketch'
    txt_path = '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/all.txt'
    fp = FileProcessor(root_dir=root_dir, extension='.png', depth=2)
    for filepath in fp.iter_files():
        name = os.path.basename(filepath)
        path_file_utils.append_line_to_file(txt_path, name)






    


def main():
    remove_name()

if __name__ == '__main__':
    main()