for num in 2e-5 4e-5 5e-6 2e-6
do
    echo "正在执行 --lr $num"
    python train_cn.py --lr $num
done