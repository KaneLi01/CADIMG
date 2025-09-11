#!/bin/bash

CONFIG_FILE="config.json"

PLY_DIR=$(jq -r '.ply_dir' $CONFIG_FILE)
HDF5_DIR=$(jq -r '.hdf5_dir' $CONFIG_FILE)
IMG_OUTPUT_DIR=$(jq -r '.img_output_dir' $CONFIG_FILE)

PLY_DIR_PARENT=$(jq -r '.ply_dir_parent' $CONFIG_FILE)
VIEW_PATH=$(jq -r '.view_path' $CONFIG_FILE)
CENTER_PATH=$(jq -r '.center_path' $CONFIG_FILE)
OUT_DIR=$(jq -r '.out_dir' $CONFIG_FILE)
CORRECT_INDEX=$(jq -r '.correct_index' $CONFIG_FILE)


for file in "$PLY_DIR"/*.ply; do
# for file in "$PLY_DIR"/00025360.ply; do
    file_name=$(basename "$file" .ply)
    echo "Processing file: $file_name"
    for SHAPE_TYPE  in 'base' 'sketch' 'target'; do
    # for SHAPE_TYPE in 'result'; do
    # 渲染三种不同形状

        DIR="${HDF5_DIR}/${SHAPE_TYPE}/${file_name}/"
        OUTPUT_DIR="${IMG_OUTPUT_DIR}${SHAPE_TYPE}/${file_name}"

        # 渲染h5文件
        blenderproc run render_normal.py --shape_type "${SHAPE_TYPE}" --ply_name "${file_name}" \
        --view_path "${VIEW_PATH}" --center_path "${CENTER_PATH}" --ply_dir "${PLY_DIR_PARENT}" --output_dir "${OUT_DIR}" --correct_index "${CORRECT_INDEX}"
        # 从h5文件中渲染图片
        # blenderproc vis hdf5 "${DIR}0.hdf5" "${DIR}1.hdf5" "${DIR}2.hdf5"  "${DIR}3.hdf5" "${DIR}4.hdf5"  "${DIR}5.hdf5" --keys "normals" --save "${OUTPUT_DIR}"


        # python /home/lkh/siga/CADIMG/utils/file_util.py --path "${OUTPUT_DIR}"
    done
done
