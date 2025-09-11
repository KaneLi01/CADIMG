import os
import utils.vis.render_cad as render_cad
import utils.cadlib.Brep_utils as Brep_utils
import utils.vis.camera_utils as camera_utils
import utils.path_file_utils as path_file_utils

def render(interrupt='00000000'):
    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'
    view_txt = '/home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/render/src/view.txt'

    base_root_dir = os.path.join(root_dir, 'base')
    add_root_dir = os.path.join(root_dir, 'add')

    base_path_list = sorted(os.listdir(base_root_dir))

    for base_path_rela in base_path_list:
        name = base_path_rela.split('.')[0]
        if name <= interrupt:
            continue
        print(f'processing {name}')
        base_path_abs = os.path.join(base_root_dir, base_path_rela)
        add_path_abs = os.path.join(add_root_dir, base_path_rela)     
        base_shape = Brep_utils.get_BRep_from_step(base_path_abs)
        add_shape = Brep_utils.get_BRep_from_step(add_path_abs)

        base_bbox = Brep_utils.get_bbox(base_shape)
        add_bbox = Brep_utils.get_bbox(add_shape)

        base_center, add_center = base_bbox.center, add_bbox.center
        
        view = camera_utils.get_view_axis(base_center, add_center, thre=0.0)
        path_file_utils.append_line_to_file(view_txt, f'({name}, {view})')


def main():
    render(interrupt='00000000')


if __name__ == '__main__':
    main()