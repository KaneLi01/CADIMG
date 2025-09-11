import os
import shutil
from typing import List, Dict


def ensure_dirs(imgs_dataset_dir: Dict[str, str], split_dirs: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    为每个 split 和 type_name 创建对应的目录，并返回路径映射字典。

    Returns:
        dict: {split: {type_name: split_type_dir}}
    """
    result = {}
    for split_name, root_dir in split_dirs.items():
        result[split_name] = {}
        for type_name in imgs_dataset_dir.keys():
            split_type_dir = os.path.join(root_dir, type_name)
            os.makedirs(split_type_dir, exist_ok=True)
            result[split_name][type_name] = split_type_dir
    return result


def copy_images(names: List[str], src_root: str, dst_dir: str) -> None:
    """
    根据文件名列表，从 src_root 拷贝到 dst_dir。

    Args:
        names (List[str]): 文件名列表
        src_root (str): 源目录
        dst_dir (str): 目标目录
    """
    for name in names:
        src_path = os.path.join(src_root, name[:8], name)
        shutil.copy(src_path, dst_dir)

def build_dataset(imgs_dataset_dir: Dict[str, str], splits: Dict[str, List[str]], split_root_dirs) -> None:
    """
    将数据集按 train/val/test 划分并拷贝到目标路径。
    """
    split_dirs = ensure_dirs(imgs_dataset_dir, split_root_dirs)
    for type_name, src_root in imgs_dataset_dir.items():
        print(f'=====processing {type_name}=====')
        for split_name, names in splits.items():
            print(f'-----processing {split_name}-----')
            dst_dir = split_dirs[split_name][type_name]
            copy_images(names, src_root, dst_dir)

