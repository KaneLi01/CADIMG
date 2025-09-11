import argparse
import yaml

def get_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "-r", "--r", 
        nargs="+",    
        help="List of r values, e.g. -r 00020841 00199999"
    )
    return parser.parse_args()


def parse_config(cfg_file):

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 必须要有的参数
    required_keys = ['task', 'task_name', 'input_root_dir', 'output_dir', 'view_path']
    
    paras_dict = {key: cfg[key] for key in required_keys}
  
    return paras_dict