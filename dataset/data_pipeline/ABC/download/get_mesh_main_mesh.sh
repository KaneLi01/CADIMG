# 下载
#URL_FILE="/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/mesh/obj_links.txt" 
URL_FILE="/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/mesh/obj_links.txt"
SAVE_DIR="/home/lkh/siga/dataset/ABC/obj_7z"
DEST_DIR="/home/lkh/siga/dataset/ABC/obj"  # 解压路径
TOTAL_NAME_LIST='/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.txt'

mkdir -p "$SAVE_DIR"
mkdir -p "$DEST_DIR"

while read -r line; do
    echo "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
    URL=$(echo "$line" | awk '{print $1}')
    FILE_NAME=$(echo "$line" | awk '{print $2}')
    NAME=$(echo "$FILE_NAME" | cut -d'_' -f2)

    echo "Downloading $FILE_NAME from $URL ..."
    
    # 下载单个chunk
    ./download_one_obj_chunk.sh "$URL" "$FILE_NAME" "$SAVE_DIR"

    echo "Unziping $FILE_NAME"
    ./unzip_valid.sh "$TOTAL_NAME_LIST" "$NAME" "$SAVE_DIR/$FILE_NAME"

    echo "Delete $FILE_NAME"
    rm "$SAVE_DIR/$FILE_NAME"  # 删除压缩包
    echo "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

done < "$URL_FILE"