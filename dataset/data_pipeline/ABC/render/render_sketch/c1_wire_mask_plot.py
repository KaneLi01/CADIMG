import os, ast
from PIL import Image

from render_cad_sketch import SketchOffscreenRenderer
from img_evaluation import cal_bw_ratio, check_border_w

from utils.rootdir_processer import FileProcessor 
import utils.cadlib.Brep_utils as Brep_utils
import utils.img_utils as img_utils
import utils.path_file_utils as path_file_utils
import utils.vis.camera_utils as camera_utils


def plot_sketch(shape_path, output_path, cam_pos, see_at, scale):
    '''给定shape路径，相机位姿，缩放尺度，输出路径，绘制sketch图像'''
    shape = Brep_utils.trans_to_ori(Brep_utils.get_BRep_from_step(shape_path))

    # 分解原始路径
    dir_path = os.path.dirname(output_path)  # 获取目录路径
    filename = os.path.basename(output_path)  # 获取文件名
    os.makedirs(dir_path, exist_ok=True)

    parts = dir_path.split(os.sep)
    if parts:  
        parts[-2] = 'intermediate_imgs'  # 中间图片路径
        name = parts[-1]
    intermediate_output_root_dir = os.sep.join(parts[:-1])

    # 输出路径
    render_configs = {
        'dash': 'render_dash_wires',
        'solid': 'render_shape_with_wires', 
        'binary': 'render_shape_black'
    }
    images = {}
    sor = SketchOffscreenRenderer()

    for style_name, render_method in render_configs.items():
        output_dir = os.path.join(intermediate_output_root_dir, style_name, name)
        os.makedirs(output_dir, exist_ok=True)
        inter_output_path = os.path.join(output_dir, filename)
        
        # 渲染
        getattr(sor, render_method)(
            output_path=inter_output_path, 
            shape=shape, 
            scale=scale, 
            cam_pos=cam_pos, 
            see_at=see_at
        )

        images[style_name] = Image.open(inter_output_path)

    # 合成图像
    img3_contour = img_utils.find_black_white_edges(images['binary']) 
    stack_img1 = img_utils.stack_imgs(images['dash'], images['solid'])
    img_utils.stack_imgs(stack_img1, img3_contour, output_path=output_path)

    return images['binary']

def pipeline(interrupt='00000000'):
    step_dir = '/home/lkh/siga/dataset/ABC/new/c1/base_add/1/step/add'
    output_dir = '/home/lkh/siga/dataset/ABC/new/c1/imgs/add_sketch/sketch'
    view_path = '/home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/render/src/view.txt'
    
    # 获取所有view
    view_dict = {}
    view_list = path_file_utils.read_txt_lines(view_path)
    for view in view_list:
        name = view.split(',')[0][1:]
        v = view.split(',')[1][1:-1]
        view_dict[name] = v

    # 绘制所有step文件
    step_list = sorted(os.listdir(step_dir))
    for step_path_rela in step_list:
        name = step_path_rela.split('.')[0]
        if name < interrupt:
            continue
        print(f'----------plotting {name}----------')
        step_path_abs = os.path.join(step_dir, step_path_rela)

        # 获取视角列表
        view = view_dict[name]
        cam_pos_list = camera_utils.compute_cam_pos_8(see_at=(0.0, 0.0, 0.0), bbox_size=0.75, view=view, scale=2)

        # 定义控制参数
        scale = 1000
        flag = 0
        whether_larger = 1

        # 多次生成以获得最佳sketch
        for j in range(30):
            for i, cam_pos in enumerate(cam_pos_list):
                output_path = os.path.join(output_dir, name, step_path_rela.split('.')[0] + f'_{i}.png')

                img_binary = plot_sketch(shape_path=step_path_abs, 
                                output_path=output_path, 
                                cam_pos=cam_pos, 
                                see_at=[0.0, 0.0, 0.0], 
                                scale=scale)    
                if not check_border_w(img_binary):
                    scale = scale - 200
                    flag = 0
                    whether_larger = 0
                    break
                else:
                    if cal_bw_ratio(img_binary) < 0.15 and whether_larger:
                        scale = scale + 200
                        flag = 0
                        break
                    else:
                        flag = 1
            if flag:
                break
       

def main():

    pipeline(interrupt='00011549')
    

    
    # plot_sketch(shape_path='/home/lkh/siga/dataset/ABC/step/01/00010019/00010019_4e3281f2d3864f2696f1e298_step_000.step', 
    #             output_path='/home/lkh/siga/dataset/ABC/step_imgs/sketch/00000006.png', 
    #             cam_pos=[2.0, 2.0, 2.0], 
    #             see_at=[0.0, 0.0, 0.0], 
    #             scale=500)

if __name__ == '__main__':
    main()