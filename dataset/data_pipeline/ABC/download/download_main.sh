# txt文件最后一行要有换行符

if [ "$1" == "all" ]; then
    URL_FILE="/home/lkh/siga/CADIMG/dataset/download/ABC/src/all_links_00.txt"  # 下载所有类型，00
    SAVE_DIR="/home/lkh/siga/dataset/ABC/all/zip"
elif [ "$1" == "step" ]; then
    URL_FILE="/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/download_data/src/step_links_raw.txt"  # 下载所有类型，00
    SAVE_DIR="/home/lkh/siga/dataset/ABC/all_7z"
elif [ "$1" == "obj" ]; then
    URL_FILE="/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/obj_links_raw.txt"  # 下载所有类型，00
    SAVE_DIR="/home/lkh/siga/dataset/ABC/7z/obj"
else 
    echo
    exit 1
fi

while read -r line; do
    URL=$(echo "$line" | awk '{print $1}')
    FILE_NAME=$(echo "$line" | awk '{print $2}')

    echo "Downloading $FILE_NAME from $URL ..."
    
    # 下载单个chunk
    ./download_one_chunk.sh "$URL" "$FILE_NAME" "$SAVE_DIR"


done < "$URL_FILE"