

# DATA_DIR="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light"
# BASE="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/base_img/00866880_4.png"
# SKETCH="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/sketch_img/00866880_4.png"



BASE_DIR="/home/lkh/siga/output/temp/722/base"
SKETCH_DIR="/home/lkh/siga/output/temp/722/sketch0"

for BASE_PATH in "$BASE_DIR"/*.png; do
    # 提取文件名（不含路径和扩展名）
    FILENAME=$(basename "$BASE_PATH")
    N="${FILENAME%.*}"  # 去掉扩展名

    SKETCH_PATH="$SKETCH_DIR/${N}.png"

    echo "处理样本：$N"

    python /home/lkh/siga/CADIMG/experiments/scripts/cad_edit.py \
        --test_img_path "$BASE_PATH" \
        --test_sketch_path "$SKETCH_PATH" \
        --index "0721_2335" \
        --device "cuda:0" \
        --torch_dtype "torch32" \
        --res 256
done