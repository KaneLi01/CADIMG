import config_loader
from rewrite_unvalid_add_shape import write_unvalid_add_base_shapes, rewrite_step
import test_vis
import utils.path_file_utils as path_file_utils
import os

def del_from_txt(config):

    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'

    base_root_dir = os.path.join(root_dir, 'base')
    add_root_dir = os.path.join(root_dir, 'add')

    input_txt_path = config['txt_name_list']
    base_path_list = sorted(path_file_utils.read_txt_lines(input_txt_path))

    for base_path_rela in base_path_list:
        base_path_abs = os.path.join(base_root_dir, base_path_rela)
        add_path_abs = os.path.join(add_root_dir, base_path_rela)  
        os.remove(base_path_abs)
        os.remove(add_path_abs)

def run_pipeline(cfg_file: str):

    # 读取配置文件
    config = config_loader.parse_config(cfg_file)

    if config['task'] == 'record_unvalid':
        write_unvalid_add_base_shapes(config)
    elif config['task'] == 'rewrite_inter' or config['task'][0:11] == 'rewrite_far':
        rewrite_step(config)
    elif config['task'][0] == '0':
        test_vis.main(config['task'])
    elif config['task'] == 'delete':
        del_from_txt(config)
    else:
        print('wrong task')


if __name__ == "__main__":
    args = config_loader.get_config_path()

    run_pipeline(args.config)