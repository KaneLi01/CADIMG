import argparse
import yaml

def get_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    return parser.parse_args()


def parse_config(cfg_file):

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 必须要有的参数
    task_name = cfg.get("task_name", "unnamed_task")
    classify_name = cfg.get("classify_name", None)  # 分类后的文件命名列表
    render = cfg.get("render", False)

    if classify_name is None:
        raise ValueError("配置文件必须包含 logic/classify_name 字段")

    if "input" in cfg and "output_dir" in cfg:
        if isinstance(cfg["input"], str):
            # 单文件情况：input是str，output_dir也必须是str
            if not isinstance(cfg["output_dir"], str):
                raise ValueError("当input为字符串时，output_dir也必须是字符串")
            
            return {
                "task": task_name,
                "input": [cfg["input"]],
                "output_dir": [cfg["output_dir"]],
                "classify_name": classify_name,
                "render": render
            }
        
        elif isinstance(cfg["input"], list):
            # 多文件情况：input是list，output_dir也必须是list且长度相同
            if not isinstance(cfg["output_dir"], list):
                raise ValueError("当input为列表时，output_dir也必须是列表")
            
            if len(cfg["input"]) != len(cfg["output_dir"]):
                raise ValueError("input列表和output_dir列表的长度必须相同")
            
            # 检查列表中的每个元素都是字符串
            if not all(isinstance(item, str) for item in cfg["input"]):
                raise ValueError("input列表中的所有元素都必须是字符串")
            
            if not all(isinstance(item, str) for item in cfg["output_dir"]):
                raise ValueError("output_dir列表中的所有元素都必须是字符串")
            
            return {
                "task": task_name,
                "input": cfg["input"],   # list
                "output_dir": cfg["output_dir"],  # 对应长度的列表
                "classify_name": classify_name,
                "render": render
            }
        
        else:
            raise ValueError("input必须是字符串或字符串列表")

    raise ValueError("配置文件必须包含 input/output_dir")