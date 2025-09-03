import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.rootdir_processer import FileProcessor
from utils.jsonl_utils import JsonlOperator
from utils.path_file_utils import write_list_to_txt, read_txt_lines



'''从step root文件夹中总结信息，用于处理在不同的del操作后重新统计信息'''

class StepDir(FileProcessor):
    '''把step文件的不同类型的信息写入jsonl文件'''
    def __init__(self, root_dir: str, feat_type='init'):
        super().__init__(root_dir, extension=".step")
    
    def process_file(self):
        pass


def get_total_jsonl():
    name_txt = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/name_total.txt'
    name_list = read_txt_lines(name_txt)

    for i in range(1, 4):
        result = []

        jsonl_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_{i}_face_edge_pro_all.jsonl'
        output_path = f'/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_{i}_final1.jsonl'
        jop = JsonlOperator(jsonl_path)
        jsonl_list = jop.load_to_list()

        for feat in jsonl_list:
            if feat['name'] in name_list:
                result.append(feat)
        
        jopw = JsonlOperator(output_path)
        jopw.write_list(result)


def get_name_list_txt():
    sd = StepDir('/home/lkh/siga/dataset/ABC/step')
    name_list = sd.list_all_second_level_subdirs()
    txt_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/name_total.txt'
    write_list_to_txt(name_list, txt_path)


def main():
    #get_name_list_txt()
    get_total_jsonl()


if __name__ == '__main__':
    main()