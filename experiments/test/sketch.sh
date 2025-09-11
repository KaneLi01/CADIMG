#!/bin/bash
NAME='00000126'
NUM=2
DIR0='/home/lkh/siga/CADIMG/experiments/test/output/'
DIR="${DIR0}""${NAME}/"
OUTPUT_DIR='/home/lkh/siga/CADIMG/experiments/test/output/imgs/'


python test_sketch_compare.py --name ${NAME} --num ${NUM} # 绘制sketch

# blenderproc run test_sketch_normal.py --name ${NAME} --num ${NUM} # 绘制sketch normal

# blenderproc vis hdf5 "${DIR}0.hdf5" --keys "normals" --save "${OUTPUT_DIR}" # 从h5文件中渲染图片

