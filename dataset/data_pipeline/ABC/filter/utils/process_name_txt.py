import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import shutil
from pathlib import Path
import argparse

import utils.cadlib.Brep_utils as Brep_utils
from utils.vis import render_cad



class Txtprocess:
    def __init__(self, file_path, root_dir, mode, output_root_dir=None, output_txt_path=None):
        self.file_path = file_path
        self.root_dir = Path(root_dir)
        self.mode = mode  # 'delete' or 'render'
        self.output_root_dir = Path(output_root_dir) if output_root_dir else None
        self.output_txt_path = output_txt_path  

    def process_file(self):
        """
        处理包含名称列表的文本文件
        """
        with open(self.file_path, 'r') as f:
            for line in f:
                name = line.strip()
                
                if not name:  # 跳过空行
                    continue
                try:
                    if self.mode == 'delete':
                        self.delete_directory(name)
                    elif self.mode == 'render':
                        self.render_shape(name)
                    elif self.mode == 'copy':
                        self.copy_name(name)
                except ValueError as e:
                    print(f"Skipping invalid name '{name}': {str(e)}")        

    def parse_name(self, name):
        """
        解析名称，从00xxyyyy格式中提取xx部分
        """
        if len(name) != 8 or not name.isdigit():
            raise ValueError(f"Invalid name format: {name}. Expected 8 digits like '00123456'")
        
        xx_part = name[2:4]
        return xx_part
    
    def get_full_path(self, name):
        """
        获取完整的目录路径
        """
        xx_part = self.parse_name(name)
        return self.root_dir / xx_part / name
    
    def get_rela_path(self, name):
        """
        获取相对路径
        """
        xx_part = self.parse_name(name)
        return Path(xx_part) / name
    
    def get_file_path(self, name):
        name_path = self.get_full_path(name)
        dir_path = Path(name_path)
        file = list(dir_path.iterdir())
        return file[0].absolute()

    
    def delete_directory(self, name):
        """
        删除指定名称对应的目录
        """
        dir_path = self.get_full_path(name)
        
        if not dir_path.exists():
            print(f"Directory does not exist: {dir_path}")
            return False
        
        try:
            shutil.rmtree(dir_path)
            print(f"Successfully deleted: {dir_path}")
            return True
        except Exception as e:
            print(f"Failed to delete {dir_path}: {str(e)}")
            return False
        
    def render_shape(self, name):
        """
        渲染指定名称对应的目录
        """
        shape_path = str(self.get_file_path(name))
        output_path = str(Path(self.output_root_dir) / f"{name}.png")
        if os.path.isfile(output_path):
            return
        print(f'processing {name}')
        shape = Brep_utils.get_BRep_from_step(shape_path)
        render_cad.save_BRep(output_path, shape=shape, bg_color=1.0)
        
    def copy_name(self, name):
        """
        复制指定名称到输出文本文件
        """
        if self.output_txt_path is None:
            raise ValueError("Output text path is not set for copying names.")
        
        if name > '00061231':
            with open(self.output_txt_path, 'a') as f:
                f.write(name + '\n')


def delete_dir(select):
    # select = 'new1' # null_child_num | face_wire_thin | simple_face_wire | reading_step_miss | shape_bbox_volume_ratio | repeat | solid
    if select is None:
        raise Exception('select is None')
    root_dir = '/home/lkh/siga/dataset/ABC/step'
    if select == 'null_child_num':
        txt_path = '/home/lkh/siga/dataset/ABC/shape_feats/processed/new/v3/filter_face_area/1/removed.txt'
        cleaner = Txtprocess(txt_path, root_dir, mode='delete')
        cleaner.process_file()
    elif select == 'reading_step_miss':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/reading_step_miss.txt'
        cleaner = Txtprocess(txt_path, root_dir, mode='delete')
        cleaner.process_file() 
    elif select == 'new1':
        txt_path = '/home/lkh/siga/dataset/ABC/shape_feats/processed/new/v3/filter_face_area/1/removed.txt'
        cleaner = Txtprocess(txt_path, root_dir, mode='delete')
        cleaner.process_file() 
    else:
        for i in range(1,4):
            if select == 'face_wire_thin':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin.txt'            
            elif select == 'simple_face_wire':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin_simple.txt'            
            elif select == 'shape_bbox_volume_ratio':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin_simple_volume.txt'
            elif select == 'thin2':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume.txt'
            elif select == 'repeat':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume_all_repeat.txt'
            elif select == 'solid':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_facewirethin2_simple_volume_repeat_solidvalid.txt'
            elif select == 'stick':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_stick.txt'    
            elif select == 'thin3':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_thin_screw2.txt'  # thin 为5
            elif select == 'screw':
                txt_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_{i}_screw.txt'  # thin 为5

            cleaner = Txtprocess(txt_path, root_dir, mode='delete')
            cleaner.process_file()            
    

def render_shape():
    select = 'all' # face_wire_thin | simple_face_wire | shape_bbox_volume_ratio | volume_thin | all
    root_dir = '/home/lkh/siga/dataset/ABC/step'

    if select == 'face_wire_thin':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_1_facewirethin.txt'
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1'
    elif select == 'simple_face_wire':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin_simple_name0.txt'
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_simple'    
    elif select == 'shape_bbox_volume_ratio':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin_simple_volume_name.txt'
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume' 
    elif select == 'volume_thin':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin2_simple_volume_name.txt'
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_thin'  
    elif select == 'all':
        txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.txt'
        output_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'          

    os.makedirs(output_root_dir, exist_ok=True)

    renderer = Txtprocess(txt_path, root_dir, mode='render', output_root_dir=output_root_dir)
    renderer.process_file()


def copy_name():
    '''
    将满足条件的名称列表复制
    根据具体情况调整copy_name函数
    '''
    root_dir = '/home/lkh/siga/dataset/ABC/step'    
    txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin_simple_name.txt'

    copyed_txt_path = txt_path.replace('.txt', '0.txt')
    print(f"Copying names to: {copyed_txt_path}")
    copyer = Txtprocess(txt_path, root_dir, mode='copy', output_txt_path=copyed_txt_path)
    copyer.process_file()


def main(args):
    if args.mode == 'delete':
        delete_dir(select=args.select)
    elif args.mode == 'copy':
        copy_name()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="delete|copy|render")
    parser.add_argument("--select", type=str, required=True, help="")
    args = parser.parse_args()

    main(args)