import os

import utils.path_file_utils as path_file_utils
from utils.rootdir_processer import FileProcessor

IMGS_DIR = {
    'sketch': '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/sketch',
    'sketch_binary': '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary',
    'sketch_dash': '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/binary',
    'sketch_solid': '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/intermediate_imgs/solid',
    'normal_base': '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/base',
    'normal_add': '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/add',
    'normal_target': '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/target'
}

class CheckName():
    '''比较name是否相同'''
    def compare_add_sketch(self):
        root_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch'
        sketch_dir = os.path.join(root_dir, 'sketch')
        intermediate_root_dir = os.path.join(root_dir, 'intermediate_imgs')

        class_name = ['binary', 'dash', 'solid']
        for cn in class_name:
            inter_dir = os.path.join(intermediate_root_dir, cn)
            path_file_utils.compare_dirs(sketch_dir, inter_dir)


    def compare_normals(self):
        root_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals'
        add_dir = os.path.join(root_dir, 'add')

        other_name = ['base', 'target']
        for cn in other_name:
            inter_dir = os.path.join(root_dir, cn)
            path_file_utils.compare_dirs(add_dir, inter_dir)


    def compare_sketch_normal(self):
        sketch_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/sketch'
        normal_add_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/all_normals/add'
        path_file_utils.compare_dirs(sketch_dir, normal_add_dir)

class CheckNameFile():
    '''检查每个name路径下的文件数量是否正确'''
    def checkfilenum(self):
        for img_name, img_dir in IMGS_DIR.items():
            print(f'checking {img_name}')
            fp = FileProcessor(img_dir, extension='.png', depth=2)
            fp.check_subdirs_num(n=8)


def check_name():
    cn = CheckName()
    # compare_add_sketch()
    # compare_normals()
    cn.compare_sketch_normal()


def check_name_file():
    cnf = CheckNameFile()
    cnf.checkfilenum()


def main():
    check_name_file()

if __name__ == '__main__':
    main()