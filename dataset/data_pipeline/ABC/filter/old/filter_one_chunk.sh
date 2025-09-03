#!/bin/bash

UNZIP_INDEX_DIR="$1"
JSON_PATH="$2"

# 创建 JSONL 文件（如果不存在则新建）
touch "$JSON_PATH"

# 删除最后一行 
# sed -i '$d' "$JSON_PATH"

if [ -s "$JSON_PATH" ]; then
    sed -i '$d' "$JSON_PATH"
    last_line=$(tail -n 1 "$JSON_PATH")
    name=$(echo "$last_line" | jq -r '.name')
    processed_name=${name: -4}
    processed_name_num=$((10#$processed_name + 1))
else
    processed_name_num=0
fi


for subdir in "$UNZIP_INDEX_DIR"/*; do
    if [ -d "$subdir" ]; then
        NAME=$(basename "$subdir")
        SUFFIX=${NAME: -4}
        SUFFIX_NUM=$((10#$SUFFIX))
        
        if [ "$SUFFIX_NUM" -ge "$processed_name_num" ]; then
            # 获取当前目录下所有文件
            files=("$subdir"/*)    
            # 初始化计数和路径
            file_path=""
            file_count=0

            for f in "${files[@]}"; do
                if [ -f "$f" ]; then
                    file_count=$((file_count + 1))
                    file_path="$f"
                fi
            done

            if [ "$file_count" -eq 1 ]; then
                echo "Processing $NAME with 1 file: $file_path"

                if ! timeout 60s python /home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/filter/old/filter_data_ABC.py \
                        --step_path "$file_path" \
                        --json_path "$JSON_PATH"; then
                    echo "timeout: $file_path"
                    printf '{"name": "%s", "valid": false, "child_num": null, "face_num": null, "wire_num": null, "bbox_min_max": null, "bbox_center": null}\n' \
                        "$NAME" >> "$JSON_PATH"                    
                fi

            else
                echo "  Invalid → writing fallback JSONL entry for $NAME"
                printf '{"name": "%s", "valid": false, "child_num": null, "face_num": null, "wire_num": null, "bbox_min_max": null, "bbox_center": null}\n' \
                    "$NAME" >> "$JSON_PATH"
            fi

        fi

    fi


done
