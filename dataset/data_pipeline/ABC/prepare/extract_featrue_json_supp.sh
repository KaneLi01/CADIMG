
i=18

# ./unzip_one_chunk.sh "$i"

python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "2"
python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "20"
                

for i in $(seq -w 19 40); do

    ./unzip_one_chunk.sh "$i"

    python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "2"
    python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "20"
                   
    rm -rf "/home/lkh/siga/dataset/ABC/temp/step/${i}"

done

for i in $(seq -w 98 99); do

    ./unzip_one_chunk.sh "$i"

    python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "2"
    python /home/lkh/siga/CADIMG/dataset/prepare_data/ABC/extract_feature_json.py --cls_idx "$i" --child_num "20"
                   
    rm -rf "/home/lkh/siga/dataset/ABC/temp/step/${i}"

done





