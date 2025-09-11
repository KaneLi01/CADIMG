import json, tempfile
import os 
from typing import Dict, Any, Iterator, List, Optional, Callable
from abc import ABC, abstractmethod
from collections import defaultdict

class BaseJsonlHandler(ABC):
    """基础处理器"""
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def _read_lines(self) -> Iterator[tuple[int, Dict[str, Any]]]:
        """逐行读取内容"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    yield line_num, json.loads(line)
                except json.JSONDecodeError as e:
                    self._handle_error(e, line, line_num)
    
    def _handle_error(self, error: Exception, raw_line: str, line_num: int):
        """错误处理"""
        print(f"Error at line {line_num}: {error}\nRaw data: {raw_line[:200]}...")


class JsonlOperator:
    """统一管理operator"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.handler = BaseJsonlHandler(filepath)
    
    # ------------ 查询类操作 ------------
    def load_to_list(self) -> List[Dict[str, Any]]:
        """加载整个文件为字典列表"""
        return [data for _, data in self.handler._read_lines()]
    
    def load_name_to_list(self) -> List[Dict[str, Any]]:
        """加载name为列表"""
        return [data['name'] for _, data in self.handler._read_lines()]
    
    def find_dict(self, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """查找特定键值对的字典"""
        for _, data in self.handler._read_lines():
            if data.get(key) == value:
                return data
        return None

    # ------------ 写入类操作 ------------
    def write_list(self, dict_list: List[Dict[str, Any]], append: bool = False) -> None:
        """
        将字典列表写入 jsonl 文件
        :param dict_list: 字典列表
        :param append: 是否追加写入，默认覆盖
        """
        mode = 'a' if append else 'w'
        with open(self.filepath, mode, encoding='utf-8') as f:
            for item in dict_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ------------ 过滤类操作 ------------
    def filter_by_func(self, condition_func: Callable[[Dict[str, Any]], bool]) -> Dict[str, List[Dict[str, Any]]]:
        """按条件函数过滤并返回分类结果  
        Returns:
            {
                "matched": [满足条件的字典列表],
                "unmatched": [不满足条件的字典列表]
            }
        """
        result = {"matched": [], "matched_name": [], "unmatched": [], "unmatched_name": []}
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_filename = temp_file.name
            print(f"创建临时文件: {temp_filename}")
            
            try:
                for i, (line_num, data) in enumerate(self.handler._read_lines()):
                    # 记录当前处理的信息到临时文件
                    temp_file.write(f"处理第 {i+1} 行: name={data.get('name', 'N/A')}\n")
                    temp_file.flush()  # 确保立即写入
                    
                    if condition_func(data):
                        result["matched"].append(data)
                        result["matched_name"].append(data['name'])
                        temp_file.write(f"  -> 匹配成功\n")
                    else:
                        result["unmatched"].append(data)
                        result["unmatched_name"].append(data['name'])
                        temp_file.write(f"  -> 不匹配\n")
                        
            except Exception as e:
                # 发生异常时，保留临时文件用于调试
                temp_file.write(f"\n处理过程中发生异常: {e}\n")
                print(f"发生异常，临时文件保留在: {temp_filename}")
                raise
        
        # 正常完成后删除临时文件
        try:
            os.unlink(temp_filename)
            print("临时文件已删除")
        except OSError as e:
            print(f"删除临时文件失败: {e}")
        
        return result

    def classify_by_func(self, category_func: Callable[[Dict[str, Any]], str]) -> Dict[str, Dict[str, List]]:
        """
        按条件函数进行多分类
        Args:
            category_func: 输入一条数据，返回类别名（可为 str/int 等），例如：
        if age < 18:
            return "young"
        elif age < 60:
            return "adult"
        """
        result = defaultdict(lambda: {"items": [], "names": []})

        for _, data in self.handler._read_lines():
            cat = category_func(data)  # 得到类别名
            result[cat]["items"].append(data)
            result[cat]["names"].append(data.get("name"))

        return dict(result)  # 转成普通字典
    


def load_jsonl_to_list(jsonl_path):
    """打开 JSONL 文件，并返回字典列表"""
    data_list = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行解析失败: {e}")
    return data_list


def find_dic(jsonl_path, key, value):
    '''查找键值是否存在，如果存在，则返回字典'''
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if data.get(key) == value:
                    return data  
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败: {e}")
    return None  # 如果没找到


def append_dic(jsonl_path, dict):
    '''将单个字典写入jsonl文件'''
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(dict, ensure_ascii=False)  # 保留中文字符
        f.write(json_line + '\n')  # 每个字典一行


def append_dict_list(jsonl_path, dict_list):
    '''将字典列表追加写入 JSONL 文件，每个字典占一行'''
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        for item in dict_list:
            json_line = json.dumps(item, ensure_ascii=False)  # 支持中文
            f.write(json_line + '\n')


def write_dict_list(jsonl_path, dict_list):
    '''将字典列表追加写入 JSONL 文件，每个字典占一行'''
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in dict_list:
            json_line = json.dumps(item, ensure_ascii=False)  # 支持中文
            f.write(json_line + '\n')


def filter_dir(jsonl_path, condition_func=None):
    '''根据条件方程进行筛选满足条件的字典'''
    if condition_func == None:  
        # 示例
        def condition(data):
            return data.get("child_num") == 1
        condition_func = condition
    
    matched_dirs = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if condition_func(data):  
                    matched_dirs.append(data)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析错误: {e}")

    return matched_dirs


def deduplicate_jsonl(jsonl_path, output_path):
    name_dict = {}      # name -> (line_num, dict)
    duplicates = {}     # name -> list of (line_num, dict)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                print(f"[第{line_num}行错误] 不是字典类型")
                continue

            name = obj.get("name")
            if name is None:
                print(f"[第{line_num}行错误] 缺少 name 字段")
                continue

            if name in name_dict:
                # 是重复项，记录所有项
                if name not in duplicates:
                    duplicates[name] = [(name_dict[name][0], name_dict[name][1])]
                duplicates[name].append((line_num, obj))
            else:
                name_dict[name] = (line_num, obj)

        except json.JSONDecodeError as e:
            print(f"[第{line_num}行错误] JSON解析失败: {e}")

    keep_lines = set()
    deleted_lines = set()

    for name, items in duplicates.items():
        # 检查是否有任何一项包含 "valid" 字段
        has_valid_field = any("valid" in d for _, d in items)

        if has_valid_field:
            valid_items = [(ln, d) for ln, d in items if d.get("valid") is True]
            invalid_items = [(ln, d) for ln, d in items if d.get("valid") is False]

            if valid_items:
                keep_lines.add(valid_items[0][0])  # 第一个 valid=True 的
            else:
                keep_lines.add(items[0][0])  # 没有 valid=True，保留第一次

            # 明确为 valid=False 的都标记为删除
            for ln, _ in invalid_items:
                deleted_lines.add(ln)
        else:
            # 没有任何 "valid"，只保留第一条
            keep_lines.add(items[0][0])
            # 其他行标记为删除
            for ln, _ in items[1:]:
                deleted_lines.add(ln)

    # 把未重复的 name 也加入保留集合（前提是不在已删除集合中）
    all_kept_names = {ln for ln, _ in name_dict.values()} - deleted_lines
    keep_lines |= all_kept_names

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(lines, 1):
            if i in keep_lines:
                f_out.write(line.strip() + '\n')

    print(f"去重完成：保留 {len(keep_lines)} 行，删除 {len(deleted_lines)} 行")


def merge_jsonls(file_a_path, file_b_path, output_path):
    # 读取文件B的所有内容，按name建立索引
    b_data = {}
    with open(file_b_path, 'r', encoding='utf-8') as file_b:
        for line in file_b:
            try:
                item = json.loads(line)
                name = item.get('name')
                if name is not None:
                    b_data[name] = item
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in file B: {line.strip()}")
                continue
    
    # 处理文件A，仅保留在B中存在name的记录
    with open(file_a_path, 'r', encoding='utf-8') as file_a, \
         open(output_path, 'w', encoding='utf-8') as out_file:
        for line in file_a:
            try:
                a_item = json.loads(line)
                name = a_item.get('name')
                
                if name in b_data:
                    # 合并两个字典，B的数据会覆盖A中相同的键
                    merged_item = {**a_item, **b_data[name]}
                    out_file.write(json.dumps(merged_item) + '\n')
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in file A: {line.strip()}")
                continue


def sort_jsonl_by_key(input_path, output_path, key="name"):
    with open(input_path, "r") as f:
        lines = [json.loads(line) for line in f]

    sorted_lines = sorted(lines, key=lambda x: x["name"])

    with open(output_path, "w") as f:
        for item in sorted_lines:
            f.write(json.dumps(item) + "\n")


def count_names_by_cls(input_file, output_file):
    from collections import defaultdict
    # 1. 初始化计数器：{"xx": count}
    cls_counter = defaultdict(int)

    # 2. 逐行读取文件并统计
    with open(input_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                name = data["name"]
                xx = name[2:4]  # 提取第3-4位
                cls_counter[xx] += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"跳过无效行: {line.strip()}，错误: {e}")

    # 3. 将统计结果写入txt文件
    with open(output_file, "w") as f:
        for xx, count in sorted(cls_counter.items()):
            f.write(f"{xx}, {count}\n")

if __name__ == '__main__':
    a = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_feat/child_num_3_facewirethin_simple_volume_all.jsonl'
    b = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_3_solid_face_edge_pro.jsonl'
    output_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/step_feat/child_num_face_edge_pro/child_num_3_face_edge_pro_all.jsonl'
    merge_jsonls(b, a, output_path)