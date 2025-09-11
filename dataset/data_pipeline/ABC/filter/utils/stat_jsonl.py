import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.jsonl_utils as jsonl_utils


def stat_add_face_num():
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/feat_total/child_num_1/1/add.jsonl'
    jop = jsonl_utils.JsonlOperator(jsonl_path)

    feats_list = jop.load_to_list()
    face_num_count = {}

    for feat in feats_list:
        fn = feat['face_num'][0]
        if fn in face_num_count:
            face_num_count[fn] += 1  
        else:
            face_num_count[fn] = 1  
    print(face_num_count)


def main():
    stat_add_face_num()

if __name__ == '__main__':
    main()