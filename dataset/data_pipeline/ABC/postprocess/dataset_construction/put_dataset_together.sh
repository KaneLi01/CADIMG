#!/bin/bash

# 将某个目录下的所有子目录中的文件放到一个大目录中
SOURCE_DIR="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views/result"
TARGET_DIR="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views_temp/target_img"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 遍历源目录下的所有子目录
for dir in "$SOURCE_DIR"/*/; do
    # 检查是否是目录
    if [ -d "$dir" ]; then
        # 获取子目录名（不含路径和末尾的/）
        dirname=$(basename "$dir")
        
        # 遍历子目录中的所有文件
        for file in "$dir"*; do
            # 检查是否是文件（不是目录）
            if [ -f "$file" ]; then
                # 获取文件名（不含路径）
                filename=$(basename "$file")
                # 新文件名为：目录名_原文件名
                new_filename="${dirname}_${filename}"
                # 移动并重命名文件到目标目录
                mv "$file" "$TARGET_DIR/$new_filename"
            fi
        done
    fi
done
