#!/bin/bash

NAME_LIST="$1"
PREFIX="$2"  
ARCHIVE="$3"  # 压缩包路径
DEST_DIR="/home/lkh/siga/dataset/ABC/obj"

mkdir -p "$DEST_DIR"

# 从 txt 中筛选出指定前缀的 name
grep "^${PREFIX}" "$NAME_LIST" > filtered_names.txt

# 构造 7z 解压命令参数（name/name_*.obj）
FILES_TO_EXTRACT=()
while read -r name; do
    FILES_TO_EXTRACT+=("${name}/${name}_*.obj")
done < filtered_names.txt

# 执行有选择性解压
7z x "$ARCHIVE" -o"$DEST_DIR" -y "${FILES_TO_EXTRACT[@]}"
