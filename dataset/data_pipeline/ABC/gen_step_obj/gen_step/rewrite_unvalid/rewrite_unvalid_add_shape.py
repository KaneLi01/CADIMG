import os
import utils.cadlib.Brep_utils as Brep_utils
import utils.path_file_utils as path_file_utils
from dataset.process.ABC.childnum1_process import Childnum1BaseAddMerge

from OCC.Core.gp import gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

def write_unvalid_add_base_shapes(config):
    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'
    
    output_txt_inter = config['txt_output_paths']['output_txt_inter']
    output_txt_far = config['txt_output_paths']['output_txt_far']
    output_txt_error = config['txt_output_paths']['output_txt_error']
    os.makedirs(root_dir, exist_ok=True)
    base_root_dir = os.path.join(root_dir, 'base')
    add_root_dir = os.path.join(root_dir, 'add')

    if config['txt_name_list'] is not None:
        input_txt_path = config['txt_name_list']
        base_path_list = sorted(path_file_utils.read_txt_lines(input_txt_path))
    else:
        base_path_list = sorted(os.listdir(base_root_dir))

    for base_path_rela in base_path_list:
        print(f'processing {base_path_rela}')
        base_path_abs = os.path.join(base_root_dir, base_path_rela)
        add_path_abs = os.path.join(add_root_dir, base_path_rela)

        try:
            base_shape = Brep_utils.get_BRep_from_step(base_path_abs)
            add_shape = Brep_utils.get_BRep_from_step(add_path_abs)
            min_dis = Brep_utils.get_min_distance(base_shape, add_shape)
            common_vol = Brep_utils.get_common_volume(add_shape, base_shape)
            if abs(min_dis) > 1e-3:
                path_file_utils.append_line_to_file(output_txt_far, base_path_rela)
            elif common_vol > 1e-3:
                path_file_utils.append_line_to_file(output_txt_inter, base_path_rela)
        except Exception as e:
            path_file_utils.append_line_to_file(output_txt_error, base_path_rela)


def rewrite_step(config):
    name_list_path = config['txt_name_list']  # 需要重写的name列表
    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'
    view_txt = config['txt_output_paths']

    name_list = path_file_utils.read_txt_lines(name_list_path)
    base_root_dir = os.path.join(root_dir, 'base')
    add_root_dir = os.path.join(root_dir, 'add')

    c1bam = Childnum1BaseAddMerge()

    for name in name_list:
        try:
            print(f'processing {name}')
            base_step_path = os.path.join(base_root_dir, name)
            add_step_path = os.path.join(add_root_dir, name)
            base_shape = Brep_utils.get_BRep_from_step(base_step_path)
            add_shape = Brep_utils.get_BRep_from_step(add_step_path)
            if config['task'] == 'rewrite_inter':
                # 有相交的部分重写
                add_shape_new, view = c1bam.compute_trans_add_shape_view(base_shape, Brep_utils.normalize_shape(add_shape), scale_factor=0.8)
                n = name.split('.')[0]
                path_file_utils.append_line_to_file(view_txt, f'({n}, {view})')
            elif config['task'] == 'rewrite_far_cylinder':
                add_shape_new, view = c1bam.compute_trans_add_shape_view(base_shape, Brep_utils.normalize_shape(add_shape), scale_factor=0.8, thre=1.0)
                n = name.split('.')[0]
                path_file_utils.append_line_to_file(view_txt, f'({n}, {view})')                
            elif config['task'] == 'rewrite_far_sphere':
                add_shape_new = rescale_sphere(base_shape, add_shape)
            Brep_utils.save_Brep_to_step(add_shape_new, add_step_path)
        except Exception as e:
            print(e)
            continue


def rescale_sphere(base_shape, add_shape):
    add_faces = Brep_utils.get_faces(add_shape)
    _, add_faces_type = Brep_utils.get_faces_area_type(add_shape)
    if not 3 in add_faces_type:
        print('no face is sphere')
        return None
    
    valid_sphere_face = None
    min_dis = float('inf')
    for face, f_type in zip(add_faces, add_faces_type):
        if f_type == 3:
            dis = Brep_utils.get_min_distance(face, base_shape)
            if dis < min_dis:
                min_dis = dis
                valid_sphere_face = face

    shpere_center, shpere_radius = Brep_utils.get_sphere_paras_from_face(valid_sphere_face)
    s2b_dis = Brep_utils.point_to_shape_distance(shpere_center, base_shape)
    scale_factor = s2b_dis / shpere_radius

    trsf_s = gp_Trsf()
    trsf_s.SetScale(shpere_center, scale_factor)  
    add_shape_s = BRepBuilderAPI_Transform(add_shape, trsf_s, True).Shape()  # 缩放后的add shape

    return add_shape_s


def is_cylinder(base_shape, add_shape):
    min_dis = Brep_utils.get_min_distance(base_shape, add_shape)  
    if min_dis <= 1e-3:
        return False

    add_faces = Brep_utils.get_faces(add_shape)
    _, add_faces_type = Brep_utils.get_faces_area_type(add_shape)
    if not 1 in add_faces_type:
        print('no face is cylinder')
        return False

    return True     
