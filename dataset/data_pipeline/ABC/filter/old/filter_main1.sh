JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'

i=00

JSON_PATH="${JSON_DIR}/${i}.jsonl"
UNZIP_INDEX_DIR="${UNZIP_DIR}/${i}"
bash /home/lkh/siga/CADIMG/dataset/create_dataset/ABC_download/filter_one_chunk.sh "$UNZIP_INDEX_DIR" "$JSON_PATH"

