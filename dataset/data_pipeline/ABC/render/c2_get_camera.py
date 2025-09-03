import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataset.prepare_data.ABC.shape_info import CADFeature, CADFeature_Child, CADFeature_Supp, Get_Feat_From_Dict
import utils.jsonl_utils as jsonl_utils
import utils.vis.camera_utils as camera_utils



def compute_cam_paras(feat):
    '''从feat中计算view'''
    f = CADFeature.from_dict(feat)
    if f.child_num == 2:
        fs = f
    elif 3 <= f.child_num <=20:
        fs = f.merge_sub_shapes()
    fs.build_sub_shapes()
    if fs.sub_shapes[0].face_num <= fs.sub_shapes[1].face_num: 
        shape_op = fs.sub_shapes[0]
        shape_base = fs.sub_shapes[1]
    else:
        shape_op = fs.sub_shapes[1]
        shape_base = fs.sub_shapes[0]                

    base_min, base_max = shape_base.bbox_min_max[:3], shape_base.bbox_min_max[3:]
    op_min, op_max = shape_op.bbox_min_max[:3], shape_op.bbox_min_max[3:]
    view = camera_utils.get_view(base_min, base_max, op_min, op_max)
    return view


def compute_jonsl_cam(child_num=2):
    '''计算jsonl文件中所有shape的view，并记录'''
    if child_num == 2:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child2_final_simpleop.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simpleop_view.jsonl'
    if child_num == 20:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child3_20_final_simpleop.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_view.jsonl'

    feats = jsonl_utils.load_jsonl_to_list(jsonl_path=jsonl_path)
    for feat in feats:
        view = compute_cam_paras(feat)
        dic = {'name': feat['name'], 'view': view}
        jsonl_utils.append_dic(output_jsonl_path, dic)


def count_view():
    '''记录没有view的shape'''
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_view.jsonl'
    feats = jsonl_utils.load_jsonl_to_list(jsonl_path=jsonl_path)
    a=0
    for feat in feats:
        if feat['view'] is not None:
            a+=1
    print(a)


def filter_view_jsonl(child_num=2):
    ''''过滤出有view的形状'''
    if child_num == 2:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simpleop_view.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simpleop_vaildview.jsonl'
    if child_num == 20:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_view.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_vaildview.jsonl'    

    feats = jsonl_utils.load_jsonl_to_list(jsonl_path=jsonl_path)
    for feat in feats:
        temp = {}
        if feat['view'] != None:
            temp['name'] = feat['name']
            temp['view'] = feat['view']
            
            jsonl_utils.append_dic(output_jsonl_path, temp)


def merge_feat_view(child_num=2):
    if child_num == 2:
        view_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simpleop_vaildview.jsonl'
        feat_jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child2_final_simpleop.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simleop_vaildview_merged.jsonl'
    if child_num == 20:
        view_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_vaildview.jsonl'    
        feat_jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/child3_20_final_simpleop.jsonl'
        output_jsonl_path = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simleop_vaildview_merged.jsonl'
    
    jsonl_utils.merge_jsonls(feat_jsonl_path, view_jsonl_path, output_jsonl_path)


def test():
    dict = {"name": "00683568", "min_dis": 0.0, "common_volume": 0.0, "valid": True, "child_num": 3, "face_num": [6, 6, 6], "wire_num": [24, 24, 24], "bbox_min_max": [[-63.50000010000001, -38.10000010000001, -1.0000000710542735e-07, 63.50000010000001, 38.10000010000001, 76.20000010000001], [-63.5000001, -38.1000001, -1.0000000355271367e-07, 63.5000001, 38.1000001, 38.1000001], [-63.5000001, -38.1000001, -12.7000001, 63.5000001, 38.1000001, 1.0000000177635683e-07]], "bbox_center": [[0.0, 0.0, 38.1], [0.0, 0.0, 19.05], [0.0, 0.0, -6.35]]}

    compute_jonsl_cam(child_num=20)


if __name__ == "__main__":
    # filter_view_jsonl(child_num=20)
    # compute_jonsl_cam(child_num=2)
    merge_feat_view(child_num=20)

