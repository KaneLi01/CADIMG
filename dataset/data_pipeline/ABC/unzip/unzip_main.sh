UNZIP_DIR='/home/lkh/siga/dataset/ABC/step'
mkdir -p "$TARGET_DIR"

# for i in $(seq -w 00 99); do

#     # 解压数据集
#     ./unzip_one_chunk.sh "$i"


# done

i=99
./unzip_one_chunk.sh "$i"