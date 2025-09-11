#!/bin/bash

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <input_file>"
    echo "支持的文件格式: .jsonl 或 .txt"
    exit 1
fi

input_file="$1"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "错误: 文件 '$input_file' 不存在"
    exit 1
fi

# 获取文件扩展名
extension="${input_file##*.}"

echo "正在处理文件: $input_file (格式: $extension)"

# 根据文件扩展名选择处理方式
case "$extension" in
    "jsonl")
        echo "检测到JSONL格式，使用JSON解析方式..."
        # 处理jsonl文件
        while read -r json_data; do
            # 跳过空行
            if [[ -z "$json_data" ]]; then
                continue
            fi
            
            name=$(jq -r '.name' <<< "$json_data")
            
            # 检查jq是否成功解析
            if [[ "$name" == "null" || -z "$name" ]]; then
                echo "警告: 无法从JSON中提取name字段: $json_data"
                continue
            fi
            
            index=${name:2:2}  # 提取第3-4位作为INDEX
            
            echo "处理: name=$name, index=$index"
            # 调用解压脚本
            ./unzip_one_chunk_name.sh "$index" "$name"
            
        done < "$input_file"
        ;;
        
    "txt")
        echo "检测到TXT格式，使用逐行读取方式..."
        # 处理txt文件
        while IFS= read -r name; do
            # 跳过空行
            if [[ -z "$name" ]]; then
                continue
            fi
            
            # 去除可能的前后空格
            name=$(echo "$name" | xargs)
            
            index=${name:2:2}  # 提取第3-4位作为INDEX
            
            echo "处理: name=$name, index=$index"
            # 调用解压脚本
            ./unzip_one_chunk_name.sh "$index" "$name"
            
        done < "$input_file"
        ;;
        
    *)
        echo "错误: 不支持的文件格式 '.$extension'"
        echo "支持的格式: .jsonl, .txt"
        exit 1
        ;;
esac


# input_file="/home/lkh/siga/CADIMG/dataset/render/ABC/src/child3_20_final_simpleop_vaildview.jsonl"

# # 逐行处理并调用解压逻辑
# while read -r json_data; do
#     name=$(jq -r '.name' <<< "$json_data")
#     index=${name:2:2}  # 提取第3-4位作为INDEX
    
#     # 这里调用你之前的解压脚本，传递变量
#     ./unzip_one_chunk_name.sh  "$index" "$name"
    
# done < "$input_file"