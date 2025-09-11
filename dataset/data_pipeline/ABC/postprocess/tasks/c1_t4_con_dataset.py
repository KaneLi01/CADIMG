import os

import utils.path_file_utils as path_file_utils
from src.c1_path import IMGS_DATASET_DIR
import dataset_construction.dataset_extract_divide as ded


# 根目录
DATASET_ROOT = "/home/lkh/siga/dataset/my_dataset/normals_train_dataset/ABC/test02"
FILTERED_TXT = "/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/record/filtered.txt"

# 读取文件名
all_names = path_file_utils.read_txt_lines(FILTERED_TXT)
splits = {
    "train": all_names[:180000],
    "val": all_names[180000:190000],
    "test": all_names[190000:200000],
}

# 目标子目录
split_root_dirs = {k: os.path.join(DATASET_ROOT, k) for k in splits.keys()}


if __name__ == "__main__":
    # ded.build_dataset(IMGS_DATASET_DIR, splits, split_root_dirs)
    ded.build_dataset({'mask_img': '/home/lkh/siga/dataset/ABC/new/c1/imgs/temp_check/normal_add/mask'}, splits, split_root_dirs)