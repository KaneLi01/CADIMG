from pathlib import Path
from PIL import Image
import os, shutil

from .condition_funcs import ImgCheck

from utils.rootdir_processer import FileProcessor
import utils.path_file_utils as path_file_utils
import utils.img_utils as img_utils

class ImageFilter(FileProcessor):
    def __init__(self, source_root: str, vis_output_dir=None, record_txt_path: str = None, depth=2, whether_filter=True):
        super().__init__(root_dir=source_root, extension='.png', depth=depth)
        self.source_root = Path(source_root)
        self.record_txt_path = Path(record_txt_path) 
        self.vis_output_dir = vis_output_dir
        self.record_txt_path.parent.mkdir(parents=True, exist_ok=True)
        os.makedirs(self.vis_output_dir, exist_ok=True)

        self.whether_filter = whether_filter
        # ic = ImgCheck()

        # self.filter_methods = {
        #     "all_border": ic.check_border,
        #     "all_single_color": ic.is_solid_color,
        #     "n_count_pixel": ic.is_small_shape,
        #     "n_2_colors": ic.is_single_color,
        #     "s_closed": ic.is_sketch_closed,
        #     "s_count_dash": ic.count_connected,
        #     "s_narrow": ic.is_narrow,
        # } 

    def filter_name_vis(self, condition_fn):
        '''需要从外部传入条件函数进行筛选'''
        for img_path in self.iter_files():
            _, name = os.path.split(img_path) 
            img = Image.open(img_path)
            if self.whether_filter:
                '''只执行过滤'''
                if condition_fn(img):
                    shutil.copy(img_path, self.vis_output_dir)
                    path_file_utils.append_line_to_file(self.record_txt_path, name)                
            else:
                path_file_utils.append_line_to_file(self.record_txt_path, f'{name}, {condition_fn(img)}')

                    
