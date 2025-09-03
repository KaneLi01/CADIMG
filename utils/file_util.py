import os
import pickle
import h5py
from PIL import Image
import cv2
import numpy as np
import trimesh
import argparse
import shutil


def compare_sets(set1, set2):

    only_in_pkl1 = set1 - set2
    only_in_pkl2 = set2 - set1

    print("文件名只在 set1 中存在：")
    for name in sorted(only_in_pkl1):
        print("  ", name)

    print("\n文件名只在 set2 中存在：")
    for name in sorted(only_in_pkl2):
        print("  ", name)

def compare_sets_same(set1, set2):
    for a in set1:
        if a in set2:
            print(a)
    
    print('---')

    for a in set2:
        if a in set1:
            print(a)

def get_set_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    keys = set(data.keys())
    print(f"pkl键数量:{len(keys)}")
    return keys


def get_set_from_dir(dir_path, file_type=None):
    files = os.listdir(dir_path)
    file_names = {f"{f.split('.')[0]}" for f in files}
    print(f"路径中文件数量:{len(file_names)}")
    return file_names


def rm_dir_without_file(dir_path):
    files = os.listdir(dir_path)
    if len(files) == 0:
        os.rmdir(dir_path)
        print(f"删除空目录: {dir_path}")


def rm_dir(dir1, dir2):
    # 获取目录1和目录2中的所有文件名
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    # 找到在目录1中但不在目录2中的文件
    only_in_dir1 = files1 - files2

    # 删除这些文件
    for file_name in only_in_dir1:
        file_path = os.path.join(dir1, file_name)
        shutil.rmtree(file_path)
        print(f"删除文件: {file_path}")




def merge_imgs(imgs_dir):
    imgs_name = sorted(os.listdir(imgs_dir)) 
    print(imgs_name)
    imgs = []
    
    for img_name in imgs_name:
        if img_name.split('.')[0] != 'total':
            img_path = os.path.join(imgs_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # ✅ RGB → BGR
            imgs.append(img_np)
    print(len(imgs))
    merged_h = np.hstack(imgs)
    cv2.imwrite(os.path.join(imgs_dir, 'total.png'), merged_h)


def from_ply_get_box(path_dir='/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/result'):
    # 读取ply文件
    center_dir = {}
    ply_files = os.listdir(path_dir)
    for ply_file in ply_files:
        ply_path = os.path.join(path_dir, ply_file)
        mesh = trimesh.load(ply_path)
        aabb = mesh.bounding_box
        min_corner = aabb.bounds[0] 
        max_corner = aabb.bounds[1]
        center = (min_corner + max_corner) /2
        center_dir[ply_file.split('.')[0]] = center
    print('finish')
    with open('/home/lkh/siga/dataset/deepcad/data/cad_ply/centers.pkl', 'wb') as f:
        pickle.dump(center_dir, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--path', type=str, help='merge imgs path')

    args = parser.parse_args()


    # merge_imgs(args.path)
    # p = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody/result'
    # paths = os.listdir(p)
    # for path in paths:
    #     path_dir = os.path.join(p, path)
    #     rm_dir_without_file(path_dir)
    path1 = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/val/base_img'
    path2 = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/val/target_img'
    # rm_dir(path2, path1)
    a = get_set_from_dir(path1)
    b = get_set_from_dir(path2)
    compare_sets(a,b)


