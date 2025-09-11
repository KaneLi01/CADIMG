import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.cadlib.Brep_utils as Brep_utils
import utils.mesh_utils as mesh_utils
import utils.vis.render_cad as render_cad
import utils.jsonl_utils as jsonl_utils
import dataset.prepare_data.ABC.shape_info as shape_info

import numpy as np
import trimesh


from OCC.Core.BRepAdaptor import BRepAdaptor_Curve


'''
将op shape中，例如profile为circle，extrude得到的侧边剔除，并将其余线条虚线化，然后输出为一个obj文件
然后渲染这个obj文件；再与渲染的真实图片进行叠加。
'''



def select_chunks(total, inter):
    '''根据点的个数和频率，选择虚线间隔的索引'''
    if total <= inter:
        return range(total)[:-1]
    idx = []

    fre = total // (inter*2) + 1  # 补齐后有多少个周期
    for n in range(fre):
        for i in range(0, inter):
            ii = n * 2 * inter + i
            if ii >= total-1:
                return idx
            idx.append(ii)
    
    return idx


def export_dottedline_mesh(wire_list, output_path, r):
    dotted_line_mesh = []
    min_distance = 1e-3

    for wire in wire_list:
        points = Brep_utils.sample_edge(wire, 32)
        if len(points) == 2:
            idx = [0]
        else:
            idx = select_chunks(len(points), 2)

        for i in idx:
            p1 = points[i]
            p2 = points[i+1]
            distance = np.linalg.norm(np.array(p2) - np.array(p1))
            if distance < min_distance:
                break
            soild_line_mesh = mesh_utils.make_cylinder(p1, p2, radius=r)
            dotted_line_mesh.append(soild_line_mesh)
    
    combined_mesh = trimesh.util.concatenate(dotted_line_mesh)
    combined_mesh.export(output_path)


def remove_extrude_edge(wire_list):
    '''line, circle, ellipse, hyperbola, parabola, beziercurve, bsplinecurve, othercurve'''

    vaild_wire_list = []
    for w in wire_list:
        curve_adaptor = BRepAdaptor_Curve(w)
        curve_type = curve_adaptor.GetType()
        if curve_type != 6:
            vaild_wire_list.append(w)

    return vaild_wire_list


def wireframe_dottedline_mask(shape, output_path):
    '''输入shape，输出为渲染虚线线框的obj'''
    lr = Brep_utils.get_scale_of_bbox(shape) / 500.0
    wire_list = Brep_utils.get_wires(shape)
    vaild_wire_list = remove_extrude_edge(wire_list)
    export_dottedline_mesh(vaild_wire_list, output_path, r=lr)


def opshape_dottedline_obj(child_num=2, dot_type=1):
    if child_num == 2:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/render_and_postprocess/ABC/src/child2_final_simleop_vaildview_merged.jsonl'
    elif child_num == 20:
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/render_and_postprocess/ABC/src/child3_20_final_simleop_vaildview_merged.jsonl'  
    elif child_num == 'supp':
        jsonl_path = '/home/lkh/siga/CADIMG/dataset/render_and_postprocess/ABC/src/temp_supp0.jsonl'

    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
    output_root_dir = '/home/lkh/siga/dataset/ABC/temp/obj'
    if dot_type == 1:
        output_root_dir = os.path.join(output_root_dir, 'dot')
    elif dot_type == -1:
        output_root_dir = os.path.join(output_root_dir, 'dot_oppo')
    cad_feats = jsonl_utils.load_jsonl_to_list(jsonl_path)

    for cad_feat in cad_feats:
        try:
            feat = shape_info.Get_Feat_From_Dict(cad_feat)
            name = feat.name
            print(f"processing {name}")
            face_num = feat.face_num

            step_cls = name[2:4]
            step_dir = os.path.join(step_root_dir, step_cls, name)
            step_file_name = os.listdir(step_dir)[0]
            step_path = os.path.join(step_dir, step_file_name)

            output_dir = os.path.join(output_root_dir, step_cls, name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, step_file_name.split('.')[0]+'.obj')

            shape = Brep_utils.get_BRep_from_step(step_path)
            sub_shapes = Brep_utils.get_child_shapes(shape)
            if child_num == 2:
                # dot_type：是否翻转op shape
                choose_first = (face_num[0] <= face_num[1]) ^ (dot_type == -1)
                shape_op = sub_shapes[0] if choose_first else sub_shapes[1]
                
            elif child_num == 20:
                choose_combined = (sum(face_num[:-1]) <= face_num[-1]) ^ (dot_type == -1)
                shape_op = Brep_utils.make_compound(sub_shapes[:-1]) if choose_combined else sub_shapes[-1]
                
            wireframe_dottedline_mask(shape_op, output_path)
            '''这里如果子形状是面或者线，会处理失败。'''
        
        except Exception as e:
            print(f"处理失败：{name}，错误：{e}")
            continue  # 跳过当前循环，继续下一个



def test():
    # 0000 3088/3521
    # /home/lkh/siga/dataset/ABC/temp/step/00/00003088/00003088_5b5eaab14b374d1992e2fe9e_step_001.step
    # /home/lkh/siga/dataset/ABC/temp/step/00/00003521/00003521_64048da97a2042849e2d9929_step_001.step
    step_path = '/home/lkh/siga/dataset/ABC/temp/step/00/00004601/00004601_976c446b3cc344ac89fb8425_step_005.step'
    output_path = '/home/lkh/siga/output/temp/wire/00004601.obj'
    shape = Brep_utils.get_BRep_from_step(step_path)
    wireframe_dottedline_mask(shape, output_path)

    # p1 = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/temp_supp.jsonl'
    # p2 = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/child2_final_simleop_vaildview_merged.jsonl'
    # o = '/home/lkh/siga/CADIMG/dataset/render/ABC/src/temp_supp0.jsonl'
    # jsonl_utils.merge_jsonls(p2, p1, o)



if __name__ == "__main__":
    opshape_dottedline_obj(child_num=2, dot_type=-1)
    # test()

