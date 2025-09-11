import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from PIL import Image
import random
from utils.img_utils import merge_imgs
import utils.path_file_utils as path_file_utils


def process_imgs(imgs_root_dir, save_root_dir, with_title=False, txt_file=None, n=None, grid_size=(3, 6)):
    """
    处理图片并按批次合并
    
    参数：
    - imgs_root_dir: str, 图片源目录
    - save_root_dir: str, 保存目录
    - with_title: bool, 是否添加标题
    - txt_file: str, 指定文件名的文本文件路径
    - n: int, 随机选择的图片数量
    - grid_size: tuple, 网格尺寸 (rows, cols)
    """
    import os
    import random
    from PIL import Image
    
    os.makedirs(save_root_dir, exist_ok=True)

    imgs_list = []
    title_list = []

    # 获取文件名列表
    if txt_file is not None:
        with open(txt_file, 'r', encoding='utf-8') as f:
            file_names = [line.strip() + '.png' for line in f if line.strip()]
    else:
        file_names = sorted([f for f in os.listdir(imgs_root_dir) 
                    if os.path.isfile(os.path.join(imgs_root_dir, f))])

    # 如果指定了数量n，随机选择
    if n is not None and n > 0:
        if n > len(file_names):
            print(f"[警告] n={n} 大于文件数量，已使用全部文件。")
        file_names = random.sample(file_names, min(n, len(file_names)))

    # 计算每批的图片数量
    rows, cols = grid_size
    batch_size = rows * cols  # 每批图片数量 = 网格容量

    for i, filename in enumerate(file_names):
        filepath = os.path.join(imgs_root_dir, filename)

        if not os.path.exists(filepath):
            print(f"找不到文件: {filepath}，跳过")
            continue

        img = Image.open(filepath)
        imgs_list.append(img)
        title_list.append(filename.split('.')[0])

        # 当达到批次大小时，执行合并
        if len(imgs_list) == batch_size:
            save_path = save_root_dir + '/' + filename
            
            if with_title:
                merge_imgs(
                    imgs_list, 
                    save_path, 
                    mode='grid', 
                    grid_size=grid_size, 
                    bg_color='white',
                    title_list=title_list
                )
            else:
                merge_imgs(
                    imgs_list, 
                    save_path, 
                    mode='grid', 
                    grid_size=grid_size, 
                    bg_color='white'
                )
            
            
            # 重置列表和批次计数
            imgs_list = []
            title_list = []

    # 处理剩余的图片（如果有的话）
    if imgs_list:
        save_path = save_root_dir + '/' + filename
        
        if with_title:
            merge_imgs(
                imgs_list, 
                save_path, 
                mode='grid', 
                grid_size=grid_size, 
                bg_color='white',
                title_list=title_list
            )
        else:
            merge_imgs(
                imgs_list, 
                save_path, 
                mode='grid', 
                grid_size=grid_size, 
                bg_color='white'
            )


def test():
    t = 7
    txt_file0 = None
    txt_file1 = None
    if t == 1:
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_simple'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_simple_merge'
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_simple_merge2'
    elif t == 2:
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_merge0'  
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_merge1'       
    elif t == 3:
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_thin'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_thin_merge0' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_volume_thin_merge1'     
    elif t == 4:
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all0' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all1' 
    elif t == 5:
        '''face thin 删掉的内容'''
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_del_facethin0' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_del_facethin1'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.txt'    
    elif t == 6:
        '''face thin 保留的内容'''      
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_facethin0' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_facethin1'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin2_simple_volume_all_repeat_facethin.txt'          
    elif t == 7:
        '''face thin2 保留的内容'''      
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_facethin20' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_facethin21'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/remain_name/child_num_1_facewirethin2_simple_volume_all_repeat_facethin2.txt'  


    process_imgs(imgs_root_dir, save_root_dir1, with_title=True, txt_file=txt_file0)
    process_imgs(save_root_dir1, save_root_dir2, with_title=False, txt_file=txt_file1)


def filter_show():
    '''展示不删除的筛选'''
    t = 'screw'
    txt_file0 = None
    txt_file1 = None
    if t == 1:
        '''对应1'''      
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/10' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/11'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_1.txt'  
    elif t == 2:   
        '''对应2'''
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/20' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/21'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/name/child_num_1_2.txt'  
    elif t == 20:   
        '''对应2的剩余部分'''
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/2ot0' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/adjust_para/2ot1'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/adjust_para/other/child_num_1_2.txt'  
    elif t == 'stick':
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/stick1' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/stick2'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_1_stick.txt' 
    elif t == 'screw':
        imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all'
        save_root_dir1 = '/home/lkh/siga/dataset/ABC/step_img_check/screw1' 
        save_root_dir2 = '/home/lkh/siga/dataset/ABC/step_img_check/screw2'   
        txt_file0 = '/home/lkh/siga/CADIMG/dataset/prepare_data/ABC/src/filter_feat/del_list/child_num_1_screw.txt'         


    process_imgs(imgs_root_dir, save_root_dir1, with_title=True, txt_file=txt_file0)
    process_imgs(save_root_dir1, save_root_dir2, with_title=False, txt_file=txt_file1)


def select_merge_imgs(
        save_dir1,
        save_dir2,
        txt_file,
        imgs_root_dir='/home/lkh/siga/dataset/ABC/step_img_check/child_num_1_all',
        n=None
        ):
    '''从filtered removed的txt列表中随机选取1620张图片进行合并；'''


    process_imgs(imgs_root_dir, save_dir1, with_title=True, txt_file=txt_file, n=n)
    process_imgs(save_dir1, save_dir2, with_title=False)


def temp_show():
    imgs_root_dir = '/home/lkh/siga/dataset/ABC/step_imgs/merge_shapes1/1/temp1'
    save_dir1 = '/home/lkh/siga/dataset/ABC/step_imgs/merge_shapes1/1/temp2'
    path_file_utils.recreate_dir(save_dir1)
    process_imgs(imgs_root_dir, save_dir1, with_title=False)
    

def main():
    temp_show()

if __name__ == "__main__":
    main()