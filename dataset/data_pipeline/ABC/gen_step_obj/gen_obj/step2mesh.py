import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.cadlib.Brep_utils as Brep_utils
import utils.vis.render_cad as render_cad
import utils.jsonl_utils as jsonl_utils
import utils.path_file_utils as path_file_utils
import utils.mesh_utils as mesh_utils


def step_2_mesh_from_jsonl():
    '''base op分别导出为mesh'''
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/ff_feats.jsonl'
    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
    output_root_dir_base = '/home/lkh/siga/dataset/ABC/temp/obj/base'
    output_root_dir_op = '/home/lkh/siga/dataset/ABC/temp/obj/operation'

    cad_feats = jsonl_utils.load_jsonl_to_list(jsonl_path)

    for cad_feat in cad_feats:
        try:
            name = cad_feat['name']
            child_num = cad_feat['child_num']
            face_num = cad_feat['face_num']
            cad_cls = name[2:4]
            print(f"processing {name}")

            output_obj_dir_base = os.path.join(output_root_dir_base, cad_cls, name)
            output_obj_dir_op = os.path.join(output_root_dir_op, cad_cls, name)
            os.makedirs(output_obj_dir_base, exist_ok=True)
            os.makedirs(output_obj_dir_op, exist_ok=True)

            step_dir = os.path.join(step_root_dir, cad_cls, name)
            step_path_rela = os.listdir(step_dir)[0]

            step_path_abs = os.path.join(step_dir, step_path_rela)
            output_obj_path_base = os.path.join(output_obj_dir_base, step_path_rela.split('.')[0]+'.obj')
            output_obj_path_op = os.path.join(output_obj_dir_op, step_path_rela.split('.')[0]+'.obj')

            shape = Brep_utils.get_BRep_from_step(step_path_abs)
            sub_shapes = Brep_utils.get_child_shapes(shape)

            if child_num == 2:
                if face_num[0] <= face_num[1]: 
                    shape_op = sub_shapes[0]
                    shape_base = sub_shapes[1]
                else: 
                    shape_op = sub_shapes[1]
                    shape_base = sub_shapes[0]
            elif 3 <= child_num <= 20:
                if sum(face_num[:-1]) <= face_num[-1]:
                    shape_op = Brep_utils.make_compound(sub_shapes[:-1])
                    shape_base = sub_shapes[-1]
                else:
                    shape_op = sub_shapes[-1]
                    shape_base = Brep_utils.make_compound(sub_shapes[:-1])

            Brep_utils.shape2mesh_save(shape_op, output_obj_path_op, linear_deflection=0.08)   
            Brep_utils.shape2mesh_save(shape_base, output_obj_path_base, linear_deflection=0.08)    
        except Exception as e: 
            print(f"{name} 出错 {e}")
            continue


def step_2_mesh():
    '''所有形状导出为mesh'''
    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
    output_root_dir = '/home/lkh/siga/dataset/ABC/temp/obj/target'
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/ff_feats.jsonl'


    for step_dir_rela in sorted(os.listdir(step_root_dir)):
        step_cls_dir = os.path.join(step_root_dir, step_dir_rela)
        output_obj_dir = os.path.join(output_root_dir, step_dir_rela)
        
        for step_dir in os.listdir(step_cls_dir):
            try:
                name = step_dir
                if jsonl_utils.find_dic(jsonl_path, 'name', name):
                    step_dir_abs = os.path.join(step_cls_dir, step_dir)

                    step_path = os.path.join(step_dir_abs, os.listdir(step_dir_abs)[0])
                    
                    output_dir = os.path.join(output_obj_dir, step_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, os.listdir(step_dir_abs)[0].split('.')[0]+'.obj')

                    if os.path.exists(output_path):
                        print(f'skip {step_path}')
                        continue
                    print(f'processing {step_path}')
                    shape = Brep_utils.get_BRep_from_step(step_path)
                    Brep_utils.shape2mesh_save(shape, output_path, linear_deflection=0.08)
            except Exception as e: 
                print(f"{step_path} 出错 {e}")
                continue


def base_add_step2mesh(interrupt='00000000'):
    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'
    output_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/obj'

    base_root_dir = os.path.join(root_dir, 'base')
    add_root_dir = os.path.join(root_dir, 'add')
    base_output_dir = os.path.join(output_dir, 'base')
    add_output_dir = os.path.join(output_dir, 'add')
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(add_output_dir, exist_ok=True)

    base_path_list = sorted(os.listdir(base_root_dir))

    for base_path_rela in base_path_list:
        try:
                
            name = base_path_rela.split('.')[0]
            if name <= interrupt:
                continue
            print(f'processing {name}')
            base_path_abs = os.path.join(base_root_dir, base_path_rela)
            add_path_abs = os.path.join(add_root_dir, base_path_rela)  

            base_shape = Brep_utils.get_BRep_from_step(base_path_abs)
            add_shape = Brep_utils.get_BRep_from_step(add_path_abs)

            base_output_path = os.path.join(base_output_dir, base_path_rela.replace('step', 'obj'))       
            add_output_path = os.path.join(add_output_dir, base_path_rela.replace('step', 'obj'))   

            _, _, base_mesh = Brep_utils.shape2mesh(base_shape, linear_deflection=0.0001)  
            _, _, add_mesh = Brep_utils.shape2mesh(add_shape, linear_deflection=0.0001) 

            base_tw = mesh_utils.TrimeshWrapper(base_mesh)
            add_tw = mesh_utils.TrimeshWrapper(add_mesh)

            base_tw.fix_mesh()
            add_tw.fix_mesh()

            base_tw.save(base_output_path)
            add_tw.save(add_output_path)
        except Exception as e:
            path_file_utils.append_line_to_file('/home/lkh/siga/CADIMG/dataset/process/ABC/src/failed_shape.txt', name)

        


def test():
    step_path = '/home/lkh/siga/dataset/ABC/temp/step/99/00990557/00990557_9d7bf00cdce2de953bbaa2d9_step_000.step'
    output_path = '/home/lkh/siga/output/temp/ortho/1.obj'
    shape = Brep_utils.get_BRep_from_step(step_path)
    Brep_utils.shape2mesh_save(shape, output_path, linear_deflection=0.08)

def main():
    #step_2_mesh()
    # test()
    base_add_step2mesh()

if __name__ == '__main__':
    main()