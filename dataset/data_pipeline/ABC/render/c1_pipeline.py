import argparse, os, inspect
from typing import Any, Callable

import utils.path_file_utils as path_file_utils
import utils.vis.camera_utils as camera_utils
from img_evaluation import cal_w_ratio, check_border_w
from render_sketch.c1_wire_mask_plot import plot_sketch
from render_normal.c1_normals_plot import plot_normals
import config_loader
from render_sketch.render_cad_sketch import SketchOffscreenRenderer

class C1RenderPipeline():
    def __init__(self, input_root_dir, output_dir, view_path, process_method='sketch'):
        '''不管是sketch还是normal，只要输入add的路径作为input_root_dir'''
        self.input_root_dir = input_root_dir
        self.output_dir = output_dir
        self.view_path = view_path
        self.view_dict = self._get_view()
        self.input_list = sorted(os.listdir(self.input_root_dir))

        # 注册渲染
        self.process_registry = {}
        self._register_default_methods()

        self.current_process_method = process_method


    def _get_view(self):
        view_dict = {}
        view_list = path_file_utils.read_txt_lines(self.view_path)
        for view in view_list:
            name = view.split(',')[0][1:]
            v = view.split(',')[1][1:-1]
            view_dict[name] = v
        
        return view_dict

    def _register_default_methods(self):
        """注册默认的处理方法"""
        self.register_process_method('sketch', plot_sketch)
        self.register_process_method('normal', plot_normals)

    def _call_process_func(self, process_func: Callable, **all_params) -> Any:
        """
        通过函数签名过滤参数
        """
        # 获取函数的参数签名
        sig = inspect.signature(process_func)
        # 过滤出函数实际需要的参数
        filtered_params = {
            key: value for key, value in all_params.items() 
            if key in sig.parameters
        }
        return process_func(**filtered_params)

    def register_process_method(self, name, method):
        """
        注册处理方法
        
        Args:
            name (str): 方法名称
            method (callable): 处理方法，接受参数 (name, base_shape_nor, target_shape, view)
        """
        self.process_registry[name] = method
    
    def render_all(self, r=['00000000', '00999999']):
        if self.current_process_method not in self.process_registry:
            raise ValueError(f"Process method '{self.current_process_method}' not registered")
        
        process_func = self.process_registry[self.current_process_method] 
        sor = SketchOffscreenRenderer()

        for input_path_rela in self.input_list:
            name = input_path_rela.split('.')[0]
            if name < r[0]:
                continue
            if name > r[1]:
                return
            print(f'=========plotting {name}=========')
            input_path_abs = os.path.join(self.input_root_dir, input_path_rela)

            # 获取视角列表
            view = self.view_dict[name]
            cam_pos_list = camera_utils.compute_cam_pos_8(see_at=(0.0, 0.0, 0.0), bbox_size=0.75, view=view, scale=2.5)
        
            sketch_scale = 750
            normal_scale = 1.0
            flag = 0
            whether_larger = 1

            # 多次生成以调整大小
            for j in range(30):
                for i, cam_pos in enumerate(cam_pos_list):
                    output_path = os.path.join(self.output_dir, name, input_path_rela.split('.')[0] + f'_{i}.png')
                    
                    # sketch 和 normal所有可能的参数
                    all_params = {
                        'shape_path': input_path_abs,
                        'output_path': output_path,
                        'cam_pos': cam_pos,
                        'see_at': [0.0, 0.0, 0.0],
                        'sketch_scale': sketch_scale,
                        'normal_scale': normal_scale,
                        'sor': sor
                    }

                    imgs = self._call_process_func(process_func, **all_params)
                    img_check = imgs['binary']
                    if not check_border_w(img_check):
                        sketch_scale = sketch_scale - 200
                        normal_scale = normal_scale - 0.2
                        flag = 0
                        whether_larger = 0
                        break
                    else:
                        if cal_w_ratio(img_check) < 0.15 and whether_larger:
                            sketch_scale = sketch_scale + 200
                            normal_scale = normal_scale + 0.2
                            flag = 0
                            break
                        else:
                            flag = 1

                if flag:
                    break

def main():

    args = config_loader.get_config_path()
    paras_dict = config_loader.parse_config(args.config)

    crp = C1RenderPipeline(input_root_dir=paras_dict['input_root_dir'], 
                           output_dir=paras_dict['output_dir'],
                           view_path=paras_dict['view_path'],
                           process_method=paras_dict['task'])
    
    crp.render_all(r=args.r)


if __name__ == '__main__':
    main()

