import argparse
import yaml

def get_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    return parser.parse_args()


def parse_config(cfg_file):
 
    if cfg_file[0] == '0':
        return {
            'task': cfg_file
        }

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 必须要有的参数
    task_name = cfg.get("task_name", None)
    task = cfg.get("task", None)

    return {
        "task": task,
        "task_name": task_name,
        "txt_name_list": cfg["txt_name_list"],
        "txt_output_paths": cfg["txt_output_paths"]          
    }



