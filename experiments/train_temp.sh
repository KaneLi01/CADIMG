# for num in 2e-2 5e-3 
# do
#     python train_cn.py --weight_decay $num --batch_size 12 --lr 1e-5 --num_epochs 15 --torch_dtype float16
# done

DATA_DIR="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/ABC/test02"
# DATA_DIR="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light"

python /home/lkh/siga/CADIMG/experiments/scripts/train_cn.py\
        --file_path $DATA_DIR\
        --tip canny\
        --torch_dtype float32\
        --batch_size 6\
        --res 256\
        --num_epochs 35 \
        --lr 3e-5 \
        # --resume_ckpt_path /home/lkh/siga/output/log/0910_1627/ckpt/controlnet_epoch1.pth
