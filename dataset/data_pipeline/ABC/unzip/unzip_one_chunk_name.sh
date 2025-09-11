# #!/bin/bash

INDEX="$1"        # 如 00
SUBDIR="$2"       # 如 00000007

DIR="/home/lkh/siga/dataset/ABC/all_7z"
FILE_PATH="${DIR}/abc_00${INDEX}_step_v00.7z"
TARGET_DIR="/home/lkh/siga/dataset/ABC/step_case/${INDEX}"
DEST_DIR="${TARGET_DIR}/${SUBDIR}"

echo "检查是否已解压：$DEST_DIR"

# 如果子目录已经存在，则跳过解压
if [ -d "$DEST_DIR" ]; then
    echo "已存在，跳过解压：$DEST_DIR"
    echo "--------------------------"
    exit 0
fi

# 若压缩文件不存在则跳过
if [ ! -f "$FILE_PATH" ]; then
    echo "压缩包不存在，跳过：$FILE_PATH"
    echo "--------------------------"
    exit 0
fi

echo "解压 ${FILE_PATH} 中的子目录 ${SUBDIR} 到 ${TARGET_DIR} ..."
mkdir -p "$TARGET_DIR"

start_time=$(date +%s)

# 解压指定子目录（包含其内部所有内容）
7z x "$FILE_PATH" "${SUBDIR}/*" -o"$TARGET_DIR" -mmt=on

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "解压 $SUBDIR 用时 ${elapsed} 秒"
echo "--------------------------"
