import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import utils.vis.camera_utils as camera_utils
import utils.jsonl_utils as jsonl_utils
import utils.mesh_utils as mesh_utils
import utils.vis.render_py3d as render_py3d
import utils.vis.render_cad as render_cad
import dataset.prepare_data.ABC.shape_info as shape_info
import trimesh
from typing import Dict, Optional
import torch

'''渲染base/target img'''
class RenderNormalImage:
    def __init__(self, cad_feat:Dict, render_obj='normal', render_num=6, render_type=1):
        """
        render_obj: 渲染normal/sketch
        render_mum: 摄像机6视角/8视角
        render_type: 渲染正向view/颠倒view和op、base
        """
        self.name = cad_feat['name']
        self.cad_cls = self.name[2:4]

        self.face_num = cad_feat['face_num']
        self.child_num = cad_feat['child_num']
        
        self.meshes = {}  # 存储加载的mesh对象、
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_valid = True
        self.render_obj = render_obj   # 'normla' | 'sketch'
        self.render_num = render_num  # 6  |  8
        self.render_type = render_type

        if render_type == 1:
            self.view = cad_feat['view']

            self.input_roots = {
                'target': '/home/lkh/siga/dataset/ABC/temp/obj/target',
                'base': '/home/lkh/siga/dataset/ABC/temp/obj/base',
                'op': '/home/lkh/siga/dataset/ABC/temp/obj/operation',
                'dot': '/home/lkh/siga/dataset/ABC/temp/obj/dot',
            }      

            self.output_roots = {
                'base': '/home/lkh/siga/dataset/ABC/temp/normal/re_base',
                'op': '/home/lkh/siga/dataset/ABC/temp/normal/re_operation',
                'dot': '/home/lkh/siga/dataset/ABC/temp/normal/re_sketch',
            }      
        elif render_type == -1:
            # 取反
            self.view = camera_utils.get_opposite_view(cad_feat['view'])

            self.input_roots = {
                'target': '/home/lkh/siga/dataset/ABC/temp/obj/target',
                'base': '/home/lkh/siga/dataset/ABC/temp/obj/base',
                'op': '/home/lkh/siga/dataset/ABC/temp/obj/operation',
                'dot': '/home/lkh/siga/dataset/ABC/temp/obj/dot_oppo',
            }      

            self.output_roots = {
                'base': '/home/lkh/siga/dataset/ABC/temp/normal/re_operation_oppo',
                'op': '/home/lkh/siga/dataset/ABC/temp/normal/re_base_oppo',
                'dot': '/home/lkh/siga/dataset/ABC/temp/normal/re_sketch_oppo',            
            }
        
        # 初始化时自动加载mesh
        self._load_meshes()

        print(f'processing {self.name}')
        self.process_mesh()

        self.cam_pos_list = self.get_cam_pos()
        self.look_at = torch.tensor([0,0,0], dtype=torch.float32, device=self.device).unsqueeze(0)

    def _load_meshes(self) -> None:
        """加载所有必需的mesh文件"""
        for key, root_dir in self.input_roots.items():
            # 构建对象目录路径
            obj_dir = os.path.join(root_dir, self.cad_cls, self.name)
            
            # 检查目录是否存在
            if not os.path.isdir(obj_dir):
                self._log_error(f"Directory not found: {obj_dir}")
                continue
                
            # 查找第一个支持的3D文件
            obj_path = os.path.join(obj_dir, os.listdir(obj_dir)[0])
            if not obj_path:
                self._log_error(f"No mesh file found in {obj_dir}")
                continue
                
            # 尝试加载mesh
            mesh = self._safe_load_mesh(obj_path)
            if mesh is not None:
                self.meshes[key] = mesh
            else: self.is_valid = False
                
    def _safe_load_mesh(self, filepath: str) -> Optional[trimesh.Trimesh]:
        """安全加载mesh文件"""
        try:
            mesh = trimesh.load(filepath)
            if not isinstance(mesh, trimesh.Trimesh):
                self._log_error(f"Loaded file is not a mesh: {filepath}")
                return None
            return mesh
        except Exception as e:
            self._log_error(f"Failed to load {filepath}: {str(e)}")
            return None        

    def _log_error(self, message: str) -> None:
        """记录错误信息并标记状态"""
        print(f"{self.name}: {message}")
        self.is_valid = False
    
    def process_mesh(self):
        """mesh标准化"""
        # 使用all mesh的标准化参数
        self.meshes_py3d = {}
        tw_target = mesh_utils.TrimeshWrapper(self.meshes['target'])
        translation, scale = tw_target.calculate_normalization_params(s=1.5)
        tw_target.apply_normalization(translation, scale)
        tw_target.fix_face_orientation() 
        # 对所有mesh应用相同的标准化参数
        self.meshes_py3d['target'] = tw_target.to_pytorch3d(self.device)  # 更新已标准化的目标mesh
        if self.render_obj == 'normal':
            l = ['base', 'op']
        elif self.render_obj == 'sketch':
            l = ['dot']
        for mesh_type in l:
            wrapper = mesh_utils.TrimeshWrapper(self.meshes[mesh_type])
            wrapper.apply_normalization(translation, scale)
            wrapper.fix_face_orientation() 
            self.meshes_py3d[mesh_type] = wrapper.to_pytorch3d(self.device)
        
        # self.meshes_py3d['target'] = mesh_utils.merge_mesh_py3d(self.meshes_py3d['op'], self.meshes_py3d['base'])
        # print(self.meshes_py3d['target'].faces_packed().shape[0], self.meshes_py3d['base'].faces_packed().shape[0], self.meshes_py3d['op'].faces_packed().shape[0])

    def get_cam_pos(self):
        if self.render_num == 6:
            return camera_utils.compute_cam_pos_mesh(self.meshes_py3d['target'], view=self.view, scale=1.0)
        elif self.render_num == 8:
            return camera_utils.compute_cam_pos_mesh_8directions(self.meshes_py3d['target'], view=self.view, scale=1.0)
    
    def render_normal(self):
        
        for i, cam_pos in enumerate(self.cam_pos_list):
            cam_pos_tensor = torch.tensor(cam_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            R, T = render_py3d.get_RT_from_cam(cam_pos_tensor, self.look_at)
            for key, mesh in self.meshes_py3d.items():
                if self.render_obj == 'normal':
                    if key in ['base', 'op']:
                        map = render_py3d.render_normal_map(mesh, device=self.device, image_size=512, R=R, T=T)
                        output_dir = os.path.join(self.output_roots[key], self.cad_cls, self.name)
                        output_path = os.path.join(output_dir, self.name+f'_{i}.png')
                        os.makedirs(output_dir, exist_ok=True)
                        render_py3d.save_png(map, output_path)
                elif self.render_obj == 'sketch':
                    if key in ['dot']:
                        map = render_py3d.render_dot_map(mesh, device=self.device, image_size=512, R=R, T=T)
                        output_dir = os.path.join(self.output_roots[key], self.cad_cls, self.name)
                        output_path = os.path.join(output_dir, self.name+f'_{i}.png')
                        os.makedirs(output_dir, exist_ok=True)
                        render_py3d.save_png(map, output_path)
    
    def render_step(self):
        '''从step文件渲染，以应用于模拟手绘sketch；同时使用不同的颜色来表示面，以便后续筛选'''
        step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'
        if self.render_type == 1:
            output_root_dir = '/home/lkh/siga/dataset/ABC/temp/normal/face'
        elif self.render_type == -1:
            output_root_dir = '/home/lkh/siga/dataset/ABC/temp/normal/face_oppo'        
        sp = shape_info.StepProcessor(step_root_dir=step_root_dir, output_root_dir=output_root_dir)
        _, output_path = sp.get_shape_outputpath(self.name)

        op_shape = sp.get_op_shape(face_num=self.face_num, child_num=self.child_num, dot_type=self.render_type)
        for i, cam_pos in enumerate(self.cam_pos_list):
            cam_pos_tensor = torch.tensor(cam_pos, dtype=torch.float32, device=self.device)
            output = output_path + f'_{i}.png'
            print(output)
            render_cad.save_BRep_ortho_face_color(shape=op_shape, output_path=output, cam_pos=cam_pos_tensor, see_at=self.look_at.squeeze())


def skip_by_exist(cad_feat, step_root_dir, output_root_dir):
    # 如果目录已存在，则不渲染
    sp = shape_info.StepProcessor(step_root_dir=step_root_dir, output_root_dir=output_root_dir)
    _, output_path = sp.get_shape_outputpath(cad_feat['name'])
    path_check = output_path + '_7.png'
    if os.path.exists(path_check):
        n = cad_feat['name']
        print(f'skip {n}')
        return True


def skip_by_txt(cad_feat, txt_path):
    target = cad_feat['name']

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip().split('/')[-1]  
            if l == target:  # 去除换行符后比较
                return False
    return True

def render():
    jsonl_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/src/filter_feat/ff_feats_sorted.jsonl'
    cad_feats = jsonl_utils.load_jsonl_to_list(jsonl_path)
    ERROR_LOG = "/home/lkh/siga/CADIMG/dataset/render_and_postprocess/ABC/src/error_log.txt"
    

    render_obj = 'sketch'  # sketch | normal
    render_num = 8
    # render_type = 1  # 1 | -1

    # 用于渲染继续
    step_root_dir = '/home/lkh/siga/dataset/ABC/temp/step'


    for render_type in [-1, 1]:
        if render_type == 1:
            re_render_txt = '/home/lkh/siga/dataset/ABC/temp/processed_imgs/check_txt/1.txt'
            output_root_dir = '/home/lkh/siga/dataset/ABC/temp/normal/re_base'
        elif render_type == -1:
            re_render_txt = '/home/lkh/siga/dataset/ABC/temp/processed_imgs/check_txt/1.txt'
            output_root_dir = '/home/lkh/siga/dataset/ABC/temp/normal/re_base_oppo'  

        for cad_feat in cad_feats:
            try:
                # # 如果目录已存在，则不渲染
                # if skip_by_exist(cad_feat, step_root_dir, output_root_dir):
                #     continue
                if skip_by_txt(cad_feat, re_render_txt):
                    continue
                rni = RenderNormalImage(cad_feat, render_obj=render_obj, render_num=render_num, render_type=render_type)
                rni.render_normal()
                # rni.render_step()
            except Exception as e:
                print(f"处理失败：{cad_feat['name']}，错误：{e}")
                with open(ERROR_LOG, "a") as f:
                    f.write(f"{cad_feat['name']}, {render_obj}, {render_type}\n")  # 只记录名称和参数
                continue  # 跳过当前循环，继续下一个


def test():
    dic = {
            "name": "00990557", 
            "min_dis": 0.0, 
            "common_volume": 0.0, 
            "valid": True, 
            "child_num": 2, 
            "face_num": [7, 6], 
            "wire_num": [32, 24], 
            "bbox_min_max": [[-18.10729122590195, -2.0072912259019486, -6.457291225901948, -13.492708774098052, 0.007291225901948476, 6.457291225901948], [-13.500000100000001, -8.400000100000002, -9.150000100000002, 13.500000100000001, 1.0000000198602732e-07, 9.150000100000002]], 
            "bbox_center": [[-15.8, -1.0, 0.0], [0.0, -4.2, 0.0]], 
            "view": "front"
        }
    rni = RenderNormalImage(dic, render_obj='sketch', render_num=8, render_type=1)


if __name__ == "__main__":
    render()