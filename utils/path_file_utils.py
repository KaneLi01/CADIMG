""" 主要包含：
    路径、文件的操作，如复制移动读写；
"""

import json, os, inspect
from pathlib import Path
import shutil
from typing import Callable


def load_json_file(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    

def load_json_get_dir(json_path, get_key='explaination', get_value='sketch single circle'):
    """
    读取一个记录字典列表的json文件，获取其中特定的值
    用于读取需要的数据集形状名称
    """
    dirs = load_json_file(json_path)
    for dir in dirs:
        if dir[get_key] == get_value:
            return dir


def get_sub_items(dir):
    """
    读取某目录下的内容，返回子目录列表或文件列表（相对）
    """
    base_path = Path(dir)
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"路径问题: {dir}")
    
    subdirs = []
    files = []

    for item in base_path.iterdir():
        if item.is_dir():
            subdirs.append(item.relative_to(base_path))
        elif item.is_file():
            files.append(item.name)

    if subdirs and files:
        raise ValueError("该目录下同时存在目录和文件")

    if subdirs:
        return [str(p) for p in sorted(subdirs)]
    elif files:
        return sorted(files)
    else:
        return []  # 空目录也返回空列表
    

def compare_dirs(dir1, dir2):
    """
    比较不同路径下的内容是否相同
    """
    dirs1 = set(get_sub_items(dir1))
    dirs2 = set(get_sub_items(dir2))
    
    only_in_dir1 = dirs1 - dirs2
    only_in_dir2 = dirs2 - dirs1

    print("\n只在第一个目录中存在的内容:")
    for path in sorted(only_in_dir1):
        print(f"- {path}")

    print("\n只在第二个目录中存在的内容:")
    for path in sorted(only_in_dir2):
        print(f"- {path}")

    return only_in_dir1, only_in_dir2


def check_subidrs_num(dir, n=6, mode='check'):
    """
    传入一个父目录，检查其所有子目录，判断每个子目录下的文件数是否满足要求
    如果传入的n=0，则检查其中没有子文件的子目录；
    如果传入的n不等于0，则检查其中文件数不等于n的子目录
    """
    subdirs = get_sub_items(dir)
    
    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        l = len(os.listdir(subdir_path))
        if not os.path.isdir(subdir_path):
            return False 

        if n == 0:
            if not os.listdir(subdir_path):  # 如果目录为空
                if mode == 'check':
                    print(f"{subdir_path}目录下没有内容")
                elif mode == 'del':
                    os.rmdir(subdir_path)  # 删除空目录
                else: raise Exception('wrong mode')
        else:
            if l != n:  
                if mode == 'check':
                    print(f"{subdir_path}目录下只有{l}个文件")
                elif mode == 'del':
                    shutil.rmtree(subdir_path)
                else: raise Exception('wrong mode')


def process_files_auto(input_root, output_root, file_handler, op_root=None, *, suffix_filter=None):
    """
    自动根据 file_handler 的参数数量处理单输入或双输入。
    """
    num_params = len(inspect.signature(file_handler).parameters)

    if num_params == 2:
        # 单输入文件处理
        for dirpath, _, filenames in os.walk(input_root):
            for filename in filenames:
                if suffix_filter and not filename.endswith(suffix_filter):
                    continue
                input_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                file_handler(input_path, output_path)

    elif num_params == 3:
        # 双输入文件处理
        assert op_root is not None, "第二输入路径 op_root 不能为空"
        for dirpath, _, filenames in os.walk(input_root):
            for filename in filenames:
                if suffix_filter and not filename.endswith(suffix_filter):
                    continue
                path1 = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(path1, input_root)
                path2 = os.path.join(op_root, rel_path)
                output_path = os.path.join(output_root, rel_path)
                if not os.path.exists(path2):
                    print(f"跳过未对齐文件: {rel_path}")
                    continue
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                file_handler(path1, path2, output_path)

    else:
        raise ValueError("file_handler 参数数量必须为2或3")


def write_filter_json(json_path, filter: str, names: list):
    '''
    将筛选的数据集name写入json文件。
    该文件是字典列表，包含两个键：筛选条件filter和通过筛选的文件名names。
    DEEPCAD数据集的两个键是 explaination 和 file_names
    '''
    dir = {'filter': filter, 'names': names}

    with open(json_path, 'r') as file:
        current_data = json.load(file)  
    current_data.append(dir)

    with open (json_path, "w") as f:
        json.dump(current_data, f, indent=4)
        f.write("\n")


def get_abs_sub_dirs(root_dir):
    """获取目录下所有子目录的绝对路径列表"""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Path is not a directory: {root_dir}")
    
    return sorted([
        os.path.join(root_dir, entry)
        for entry in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, entry))
    ])


def write_list_to_txt(data_list, filepath):
    """将列表写入文件，每个元素一行"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(f"{item}\n")  # 自动添加换行符


def read_txt_lines(filepath: str) -> list:
    """
    从txt文件中读取数据，每一行作为一个元素写入列表
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去掉首尾空白符（包括换行符）
            if line:  # 跳过空行
                lines.append(line)
    return lines


def append_line_to_file(filename, content):
    """将内容追加到文件，确保作为新的一行（如果文件不存在则创建）"""
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            # 检查文件是否为空或是否需要换行
            f.seek(0, 2)  # 移动到文件末尾
            if f.tell() > 0:  # 如果文件不为空
                f.write('\n' + content)
            else:  # 如果文件为空
                f.write(content)
        print(f"内容已成功追加到文件: {filename}")
    except Exception as e:
        print(f"写入文件时出错: {e}")


def merge_txt(file1_path, file2_path):
    # 读取 file1 的所有行
    with open(file1_path, 'r', encoding='utf-8') as f1:
        lines1 = set(line.strip() for line in f1 if line.strip())  # 去除空行和换行

    # 读取 file2 的所有行
    with open(file2_path, 'r', encoding='utf-8') as f2:
        lines2 = [line.strip() for line in f2 if line.strip()]

    # 找出 file2 中不在 file1 的新行
    new_lines = [line for line in lines2 if line not in lines1]

    # 追加这些新行到 file1
    with open(file1_path, 'a', encoding='utf-8') as f1:
        for line in new_lines:
            f1.write(line + '\n')

    print(f"已合并，添加了 {len(new_lines)} 条新记录到 {file1_path}")


def rename_file(file_path):
    """根据规则重命名单个文件"""
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    
    if len(parts) < 2 or not filename.endswith(".png"):
        return  # 跳过不符合格式的

    first_part = parts[0]
    last_part = parts[-1]  # e.g. '0.png'
    new_name = f"{first_part}_{last_part}"

    new_path = os.path.join(os.path.dirname(file_path), new_name)

    if file_path != new_path:
        print(f"重命名: {filename} -> {new_name}")
        os.rename(file_path, new_path)

def rename_recursively(root_dir):
    """递归遍历所有子文件并重命名，用于将混乱的文件名统一格式"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rename_file(full_path)


def remove_prefix(dir_a, dir_b):
# 遍历目录 B，删除不在 A 中的文件
    a_files = set(os.listdir(dir_a))
    for file in os.listdir(dir_b):
        b_file_path = os.path.join(dir_b, file)
        if os.path.isfile(b_file_path) and file not in a_files:
            os.remove(b_file_path)
            print(f"删除：{file}")


def copy_file_to_directory(source_file_path: str, target_directory: str):
    """
    将源文件复制到目标目录
    """
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"源文件不存在: {source_file_path}")
    
    if not os.path.isfile(source_file_path):
        raise ValueError(f"路径不是文件: {source_file_path}")
    
    # 确保目标目录存在
    os.makedirs(target_directory, exist_ok=True)
    
    # 获取文件名
    filename = os.path.basename(source_file_path)
    
    # 构建目标路径
    target_path = os.path.join(target_directory, filename)
    
    # 复制文件
    shutil.copy(source_file_path, target_path)

    return target_path


def recreate_dir(dir_path: str):
    """
    如果目录已存在，删除其中所有内容并重新创建
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # 删除整个目录及其内容
    os.makedirs(dir_path)        # 重新创建空目录


def sort_txt_file(input_file_path, output_file_path=None):
    """
    读取txt文件，按字符串顺序对每一行进行排序，然后写入新文件
    
    Args:
        input_file_path (str): 输入文件的路径
        output_file_path (str, optional): 输出文件的路径。如果为None，则在原文件名后加_sorted
        
    Returns:
        str: 输出文件的路径
    """
    # 读取文件内容
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"错误：文件 '{input_file_path}' 不存在")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
    # 去除每行末尾的换行符并过滤空行
    lines = [line.strip() for line in lines if line.strip()]
    
    # 按字符串顺序排序
    lines.sort()
    
    # 生成输出文件路径
    if output_file_path is None:
        import os
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_sorted.txt"
    
    # 写入排序后的内容到新文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line + '\n')
        print(f"文件已排序并保存到: {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"写入文件时发生错误: {e}")
        return None


def test():
    dir1 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/neg.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/size/neg_base_size.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/size/neg_operation_size.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/single_color/neg_base_single_color.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/single_color/neg_operation_single_color.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/narrow/neg_solid_narrow_processed.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/dot_num/neg_sketch_dot_num_processed.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/cover/neg_operation_cover.txt'
    # dir2 = '/home/lkh/siga/dataset/ABC/rough_data/record/neg/blank/neg_base_blank.txt'
    # merge_txt(dir1, dir2)

    # check_subidrs_num('/home/lkh/siga/dataset/ABC/filter_dataset/pos', 0, 'del')
    
    dir1 = '/home/lkh/siga/output/temp/722/base'
    dir2 = '/home/lkh/siga/output/temp/722/sketch0'
    compare_dirs(dir1, dir2)



if __name__ == "__main__":
    test()