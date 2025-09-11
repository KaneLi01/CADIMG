import yaml, json, datetime, argparse, os, shutil
import importlib
import basic_filter_funcs
import config_loader
from dataset.prepare_data.ABC.shape_info import Get_Feat_From_Dict
import utils.jsonl_utils as jsonl_utils  # 你项目里已有的工具
import utils.path_file_utils as path_file_utils
import utils.merge_imgs as merge_imgs

class JsonlClassify():
    '''通过child多分类后的jsonl文件，进行过滤'''
    def __init__(self, jsonl_path: str, filter_func, category_paths=None):
        self.jsonl_path = jsonl_path
        self.filter_func = filter_func
        self.category_paths = category_paths
        self.jop = jsonl_utils.JsonlOperator(self.jsonl_path)
        self.handler = self.jop.handler     

    def filter(self) -> None:
        
        cat_len_dict = {}  # 记录数据集数量
        result = self.jop.classify_by_func(self.filter_func)
        for i, (category, items) in enumerate(result.items()):

            cat_paths = self.category_paths[category]
            names = [d for d in items['names']]
            feats = [f for f in items['items']]
            cat_len_dict[category] = len(names)

            # 如果需要按类别保存到 txt
            output_txt = getattr(self, f"{category}_name_output_txt", cat_paths['txt'])
            if output_txt:
                path_file_utils.write_list_to_txt(names, output_txt)

            # 如果需要按类别保存到 jsonl
            output_jsonl = getattr(self, f"{category}_output_jsonl", cat_paths['jsonl'])
            if output_jsonl:
                jsonl_utils.write_dict_list(output_jsonl, feats)
    
        # 记录数量
        return cat_len_dict


def run_one_file(input_path, output_dir, classify_name, filter_func, task_name, render):

    # 获取保存路径
    category_paths = {}
    for category in classify_name:
        category_paths[category] = {
            "jsonl": os.path.join(output_dir, f"{category}.jsonl"),
            "txt": os.path.join(output_dir, f"{category}.txt"),
            "img_tmp": os.path.join(output_dir, 'vis', f"{category}_temp"),
            "img_final": os.path.join(output_dir, 'vis', category)            
        }
        if render:
            path_file_utils.recreate_dir(category_paths[category]["img_tmp"])
            path_file_utils.recreate_dir(category_paths[category]["img_final"])

    jc = JsonlClassify(
    jsonl_path=input_path, 
    filter_func=filter_func,
    category_paths=category_paths
    )
    cat_len_dict = jc.filter()

    if render:
        for category in classify_name:
            print(f'plot {category} imgs')
            merge_imgs.select_merge_imgs(
                save_dir1=category_paths[category]["img_tmp"],
                save_dir2=category_paths[category]["img_final"],
                txt_file=category_paths[category]["txt"],
                n=1620
            )
            shutil.rmtree(category_paths[category]["img_tmp"])
    
    return cat_len_dict  # 返回数据集大小，用于记录


def run_pipeline(cfg_file: str):

    # 读取配置文件
    task = config_loader.parse_config(cfg_file)

    # 导入task name同名函数
    module_path = 'logic'
    func_name = task['task']
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)

    for input_path, output_dir in zip(task["input"], task["output_dir"]):
        path_file_utils.recreate_dir(output_dir)
        target_path = path_file_utils.copy_file_to_directory(cfg_file, output_dir)
        cat_len_dict = run_one_file(
            input_path=input_path,
            output_dir=output_dir,
            classify_name=task["classify_name"],
            filter_func=func,
            task_name=task["task"],
            render=task['render']
        )

        len_write_data = {'cats_len':cat_len_dict}
        with open(target_path, "a", encoding="utf-8") as f:
            yaml.dump(len_write_data, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    args = config_loader.get_config_path()

    run_pipeline(args.config)
