import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.jsonl_utils as jsonl_utils

from pathlib import Path

def check_dict_valid(dic, line_num, name_index, mode='init'):

    if mode == 'init' or mode == 'filter':
        required_keys = ["name", "valid", "child_num", "face_num", "wire_num", "bbox_min_max", "bbox_center"]
    elif mode == 'supp':
        required_keys = ["name", "min_dis", "common_volume"]
    else: raise ValueError('wrong mode')

    for key in required_keys:
        if key not in dic:
            raise Exception(f"第{line_num}行缺少键：{key}")

    if mode == 'init':
        # if not f"00{name_index}0000" <= dic["name"] <= f"00{name_index}9999":
        #     raise ValueError(f"第{line_num}行的name格式错误")
        if not dic["name"] == f"00{name_index}{(line_num-1):04d}":
            print(f"00{name_index}{(line_num-1):04d}")
            raise ValueError(f"第{line_num}行的name序号错误")        
    elif mode == 'filter' or mode == 'supp':
        if not (
            isinstance(dic["name"], str)
            and len(dic["name"]) == 8 
            and dic["name"].isdigit()
            ):
            raise ValueError(f"第{line_num}行的name格式错误")
    
    if mode == 'init' or mode == 'filter':

        if not isinstance(dic["valid"], bool):
            raise ValueError(f"第{line_num}行的vaild格式错误")

        if dic["valid"]:
            child_num = dic["child_num"]
            if not (isinstance(child_num, int)):
                raise ValueError(f"第{line_num}行的child_num应为整数")
            if not (isinstance(dic["face_num"], list) and len(dic["face_num"]) == child_num and all(isinstance(x, int) for x in dic["face_num"])):
                raise ValueError(f"第{line_num}行的face_num有误")
            if not (isinstance(dic["wire_num"], list) and len(dic["wire_num"]) == child_num and all(isinstance(x, int) for x in dic["wire_num"])):
                raise ValueError(f"第{line_num}行的wire_num有误")

            if not isinstance(dic["bbox_min_max"], list):
                raise ValueError(f"第{line_num}行的 bbox_min_max 应为 list 类型")

            if len(dic["bbox_min_max"]) != child_num:
                raise ValueError(f"第{line_num}行的 bbox_min_max 长度应为 child_num={child_num}，实际为 {len(dic['bbox_min_max'])}")

            for sub_idx, sub in enumerate(dic["bbox_min_max"]):
                if not isinstance(sub, list):
                    raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}] 应为 list 类型")
                if len(sub) != 6:
                    raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}] 应有 6 个元素，实际为 {len(sub)}")
                for val_idx, val in enumerate(sub):
                    if not isinstance(val, float):
                        raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}][{val_idx}] 应为 float，实际为 {type(val)}")

            # 检查 bbox_center
            if not isinstance(dic["bbox_center"], list):
                raise ValueError(f"第{line_num}行的 bbox_center 应为 list 类型")

            if len(dic["bbox_center"]) != child_num:
                raise ValueError(f"第{line_num}行的 bbox_center 长度应为 child_num={child_num}，实际为 {len(dic['bbox_center'])}")

            for sub_idx, sub in enumerate(dic["bbox_center"]):
                if not isinstance(sub, list):
                    raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}] 应为 list 类型")
                if len(sub) != 3:
                    raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}] 应有 3 个元素，实际为 {len(sub)}")
                for val_idx, val in enumerate(sub):
                    if not isinstance(val, float):
                        raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}][{val_idx}] 应为 float，实际为 {type(val)}")

        else:
            # valid 为 False，其他字段必须为 None
            for key in ["child_num", "face_num", "wire_num", "bbox_min_max", "bbox_center"]:
                if dic[key] is not None:
                    raise ValueError(f"第{line_num}行的{key} 应为 null")
    
    elif mode == 'supp':
        if not isinstance(dic["min_dis"], float) or not isinstance(dic["common_volume"], float):
            raise ValueError(f"第{line_num}行数值错误")


def check_jsonl_vaild(jsonl_path, mode='init'):
    '''
    mode为检查的文件类型
    当mode=init时，检查初始生成的json文件。
    当mode=filter时，检查根据child num，复杂度，长宽比等调剂筛选的json文件
    '''

    name_index = jsonl_path.split('/')[-1].split('.')[0]

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"第{line_num}行不是字典类型：{type(obj)}")
                
                check_dict_valid(obj, line_num, name_index, mode=mode)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[第{line_num}行错误] {e}")
                return 0


def check_json_cls(jsonl_path):
    '''检查单个文件中，每个cls的数量以及是否齐全'''

    count = {}

    for i in range(0,100):
        cls = f"{i:04d}"
        count[cls] = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"第{line_num}行不是字典类型：{type(obj)}")
                
                name = obj.get("name")
                cls = name[:4]
                count[cls] += 1

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[第{line_num}行错误] {e}")
                return 0
    
    return count


def dedup_init_jsons():
    '''去重目录下初始筛选的json文件'''
    raw_dir = '/home/lkh/siga/dataset/ABC/temp/filter_log'
    out_dir = '/home/lkh/siga/dataset/ABC/temp/filter_log_dedup'
    files = os.listdir(raw_dir)
    for f in files:
        path = os.path.join(raw_dir, f)
        opath = os.path.join(out_dir, f)
        jsonl_utils.deduplicate_jsonl(path,opath)


def check_init_jsons():
    '''检查目录下的初始筛选的json文件是否规范'''
    raw_dir = '/home/lkh/siga/dataset/ABC/all/raw'
    files = os.listdir(raw_dir)
    for f in files:
        path = os.path.join(raw_dir, f)
        check_jsonl_vaild(path)


def check_view_unzip(child_num=2):
    '''检查根据view文件解压的step文件是否完整'''

    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
    
    cls_dir = {}
    for child_num in [2, 20]:
        if child_num == 2:
            jsonl_path = "/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simpleop_vaildview.jsonl"
        elif child_num == 20:
            jsonl_path = "/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_vaildview.jsonl"
        
        cad_feats = jsonl_utils.load_jsonl_to_list(jsonl_path)
        for cad_feat in cad_feats:
            name = cad_feat['name']

            step_cls = name[2:4]
            cls_dir[step_cls] = cls_dir.get(step_cls, 0) + 1
            step_dir = os.path.join(step_root_dir, step_cls, name)
            if not os.path.isdir(step_dir):
                print(f"{name}不存在")

            if not os.listdir(step_dir):
                print(f"{name}目录是空的")
    print(cls_dir)




def main():
    # path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child3_20_simple_boxy_supp_old.jsonl'
    # o = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/test.jsonl'
    # jsonl_utils.deduplicate_jsonl(path, o)
    check_view_unzip(20)

if __name__ == "__main__":
    main()

