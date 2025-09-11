# mesh_utils.py
import os
import torch
import trimesh
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


class TrimeshWrapper:
    def __init__(self, mesh: trimesh.Trimesh = None):
        self.mesh = mesh

    @classmethod
    def load(cls, path):
        """
        从文件加载 mesh
        """
        mesh = trimesh.load(path, process=False)
        return cls(mesh)
    
    def save(self, path):
        """保存 mesh 到文件"""
        self.mesh.export(path)

    def fix_mesh(self):
        """修复"""
        self.mesh.merge_vertices()  # 合并重复顶点
        self.mesh.remove_duplicate_faces()  # 移除重复面

        # 确保法向量朝外
        
        if not self.mesh.is_watertight:
            print("警告: 网格不是封闭的，法向量方向可能不准确")
            self.robust_mesh_repair()
        self.mesh.fix_normals()
    
    def calculate_normalization_params(self, s=1.0):
        """计算将mesh标准化到原点所需的平移和缩放参数"""
        assert self.mesh is not None, "Mesh not initialized."
        
        # 计算包围盒中心和最大边长
        bbox = self.mesh.bounding_box
        translation = -bbox.centroid  # 需要平移的向量
        scale = s / np.max(bbox.extents)  # 标准化缩放系数
        
        return translation, scale

    def apply_normalization(self, translation, target_scale=1.0):
        """平移和缩放参数到mesh"""
        assert self.mesh is not None, "Mesh not initialized."
        
        # 应用平移和缩放
        self.mesh.apply_translation(translation)
        self.mesh.apply_scale(target_scale)
        return self

    def normalization(self, target_scale=1.0):
        translation, base_scale = self.calculate_normalization_params()
        return self.apply_normalization(translation, target_scale * base_scale)

    def to_pytorch3d(self, device='cpu', flip=False):
        """
        返回 PyTorch3D 所需的 verts 和 faces
        """
        verts = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=device)
        if flip:
            faces[:, [1, 2]] = faces[:, [2, 1]]
        mesh = Meshes(verts=[verts], faces=[faces]).to(device)
        return mesh
    
    def robust_mesh_repair(self, verbose=False):
        """
        统一的mesh修复函数，解决零面积三角形和其他常见问题
        
        Args:
            verbose: 是否打印详细信息
        
        Returns:
            self: 返回自身以支持链式调用
        """
        if verbose:
            print("=== 开始Mesh修复 ===")
            print(f"原始: 顶点={len(self.mesh.vertices)}, 面={len(self.mesh.faces)}")
        
        # 1. 首先移除零面积和退化三角形
        self.remove_degenerate_triangles(verbose)
        
        # 2. 智能顶点合并（多次迭代，使用不同容差）
        self.smart_vertex_merge(verbose)
        
        # 3. 再次清理退化面
        self.remove_degenerate_triangles(verbose=False)  # 第二次不打印详细信息
        
        # 4. 修复小孔洞
        self.fix_small_holes(verbose)
        
        # 5. 最终清理和法向量修复
        self.mesh.remove_duplicate_faces()
        self.mesh.fix_normals()
        
        if verbose:
            print(f"最终: 顶点={len(self.mesh.vertices)}, 面={len(self.mesh.faces)}")
            print(f"密闭性: {self.mesh.is_watertight}")
            try:
                print(f"欧拉数: {self.mesh.euler_number}")
            except:
                print("欧拉数: 计算失败")
            print("=== 修复完成 ===\n")
        
        return self

    def remove_degenerate_triangles(self, verbose=False):
        """
        移除零面积和退化三角形
        """
        if len(self.mesh.faces) == 0:
            return self
            
        original_faces = len(self.mesh.faces)
        
        # 计算面积
        face_areas = self.mesh.area_faces
        
        # 找出零面积或接近零面积的三角形
        area_threshold = 1e-12
        degenerate_by_area = face_areas < area_threshold
        
        # 找出边长度异常的三角形（比如有边长为0）
        degenerate_by_edges = self.find_degenerate_by_edges()
        
        # 合并所有退化面
        degenerate_faces = degenerate_by_area | degenerate_by_edges
        
        if np.any(degenerate_faces):
            valid_faces = self.mesh.faces[~degenerate_faces]
            self.mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=valid_faces, process=False)
            
            if verbose:
                print(f"移除退化三角形: {np.sum(degenerate_faces)} 个 (面积<{area_threshold}: {np.sum(degenerate_by_area)}, 边长异常: {np.sum(degenerate_by_edges)})")
        
        return self

    def find_degenerate_by_edges(self):
        """
        通过边长找出退化三角形
        """
        if len(self.mesh.faces) == 0:
            return np.array([], dtype=bool)
            
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        # 计算每个三角形的三条边
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]  
        v2 = vertices[faces[:, 2]]
        
        # 计算边长
        edge1 = np.linalg.norm(v1 - v0, axis=1)
        edge2 = np.linalg.norm(v2 - v1, axis=1)
        edge3 = np.linalg.norm(v0 - v2, axis=1)
        
        # 找出有边长接近0的三角形
        edge_threshold = 1e-10
        degenerate = (edge1 < edge_threshold) | (edge2 < edge_threshold) | (edge3 < edge_threshold)
        
        return degenerate

    def smart_vertex_merge(self, verbose=False):
        """
        智能顶点合并，使用多种策略
        """
        original_vertices = len(self.mesh.vertices)
        
        # 策略1: 使用trimesh默认合并
        self.mesh.merge_vertices()
        
        if verbose:
            print(f"默认合并: {original_vertices} -> {len(self.mesh.vertices)} 顶点")
        
        # 策略2: 如果还有问题，使用更激进的合并
        if not self.mesh.is_watertight:
            self.aggressive_vertex_merge(verbose)
        
        return self

    def aggressive_vertex_merge(self, verbose=False, tolerance_factors=[1e-6, 1e-5, 1e-4]):
        """
        更激进的顶点合并策略
        """
        best_mesh = self.mesh.copy()
        best_score = self.calculate_mesh_quality_score(best_mesh)
        
        for tol_factor in tolerance_factors:
            # 计算基于mesh尺寸的容差
            bbox_size = np.max(self.mesh.bounds[1] - self.mesh.bounds[0])
            tolerance = bbox_size * tol_factor
            
            test_mesh = self.mesh.copy()
            
            # 手动合并接近的顶点
            vertex_map = self.find_close_vertices(test_mesh.vertices, tolerance)
            
            if len(vertex_map) > 0:
                new_vertices, new_faces = self.apply_vertex_mapping(test_mesh.vertices, test_mesh.faces, vertex_map)
                test_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                
                # 清理
                test_mesh.remove_duplicate_faces()
                test_mesh = self.remove_degenerate_triangles_static(test_mesh)
                
                # 评估质量
                score = self.calculate_mesh_quality_score(test_mesh)
                
                if verbose:
                    print(f"容差 {tolerance:.2e}: 质量分数 {score:.3f}, 密闭: {test_mesh.is_watertight}")
                
                if score > best_score:
                    best_mesh = test_mesh
                    best_score = score
        
        self.mesh = best_mesh
        return self

    def find_close_vertices(self, vertices, tolerance):
        """
        找出距离小于容差的顶点对
        """
        vertex_map = {}
        n_vertices = len(vertices)
        
        # 使用简化的方法找近邻
        for i in range(n_vertices):
            if i in vertex_map:
                continue
                
            # 找出与当前顶点距离小于容差的其他顶点
            distances = np.linalg.norm(vertices - vertices[i], axis=1)
            close_indices = np.where((distances < tolerance) & (distances > 0))[0]
            
            # 将这些顶点映射到当前顶点
            for j in close_indices:
                if j not in vertex_map:
                    vertex_map[j] = i
        
        return vertex_map

    def apply_vertex_mapping(self, vertices, faces, vertex_map):
        """
        应用顶点映射，合并顶点
        """
        # 创建新的顶点索引映射
        n_vertices = len(vertices)
        new_index_map = np.arange(n_vertices)
        
        # 应用合并映射
        for old_idx, new_idx in vertex_map.items():
            new_index_map[old_idx] = new_idx
        
        # 重新索引面
        new_faces = new_index_map[faces]
        
        # 创建唯一顶点列表
        unique_indices = np.unique(new_index_map)
        vertex_reindex = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        final_vertices = vertices[unique_indices]
        final_faces = np.array([[vertex_reindex[new_faces[i, j]] for j in range(3)] for i in range(len(new_faces))])
        
        return final_vertices, final_faces

    def fix_small_holes(self, verbose=False):
        """
        修复小孔洞
        """
        if self.mesh.is_watertight:
            return self
        
        original_watertight = self.mesh.is_watertight
        
        # 尝试填充孔洞
        try:
            self.mesh.fill_holes()
            if verbose and not original_watertight and self.mesh.is_watertight:
                print("成功填充孔洞，mesh现在密闭")
        except Exception as e:
            if verbose:
                print(f"填充孔洞时出错: {e}")
        
        return self

    def calculate_mesh_quality_score(self, mesh):
        """
        计算mesh质量分数（越高越好）
        """
        score = 0
        
        # 密闭性（最重要）
        if mesh.is_watertight:
            score += 10
        
        # 面积分布（避免零面积面）
        face_areas = mesh.area_faces
        if len(face_areas) > 0:
            min_area = np.min(face_areas)
            if min_area > 1e-12:
                score += 5
            
            # 面积分布的一致性
            area_cv = np.std(face_areas) / np.mean(face_areas) if np.mean(face_areas) > 0 else float('inf')
            if area_cv < 2.0:  # 变异系数较小
                score += 2
        
        # 欧拉数检查（对于简单封闭形状应该是2）
        try:
            euler_num = mesh.euler_number
            if euler_num == 2:
                score += 3
        except:
            pass
        
        return score

    @staticmethod
    def remove_degenerate_triangles_static(mesh, verbose=False):
        """
        静态方法：移除零面积和退化三角形（用于内部处理）
        """
        if len(mesh.faces) == 0:
            return mesh
            
        # 计算面积
        face_areas = mesh.area_faces
        
        # 找出零面积或接近零面积的三角形
        area_threshold = 1e-12
        degenerate_by_area = face_areas < area_threshold
        
        # 找出边长度异常的三角形
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 计算每个三角形的三条边
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]  
        v2 = vertices[faces[:, 2]]
        
        # 计算边长
        edge1 = np.linalg.norm(v1 - v0, axis=1)
        edge2 = np.linalg.norm(v2 - v1, axis=1)
        edge3 = np.linalg.norm(v0 - v2, axis=1)
        
        # 找出有边长接近0的三角形
        edge_threshold = 1e-10
        degenerate_by_edges = (edge1 < edge_threshold) | (edge2 < edge_threshold) | (edge3 < edge_threshold)
        
        # 合并所有退化面
        degenerate_faces = degenerate_by_area | degenerate_by_edges
        
        if np.any(degenerate_faces):
            valid_faces = mesh.faces[~degenerate_faces]
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=valid_faces, process=False)
        
        return mesh

    @staticmethod
    def from_pytorch3d(mesh: Meshes):
        """
        从 PyTorch3D 的 Meshes 创建 TrimeshWrapper
        """
        verts = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()
        mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return TrimeshWrapper(mesh_tri)


def make_cylinder(p1, p2, radius=0.1, sections=16):
    """创建两个点之间的圆柱体（三角网格）"""
    vec = np.array(p2) - np.array(p1)
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return None

    direction = vec / height
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cylinder.apply_translation([0, 0, height / 2.0])

    axis = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(axis)
    cylinder.apply_translation(p1)
    return cylinder


def merge_mesh_py3d(mesh1, mesh2):
    verts1 = mesh1.verts_packed()  # (V1, 3)
    faces1 = mesh1.faces_packed()  # (F1, 3)
    verts2 = mesh2.verts_packed()  # (V2, 3)
    faces2 = mesh2.faces_packed()  # (F2, 3)

    # 合并顶点和面（注意面的索引需要偏移）
    combined_verts = torch.cat([verts1, verts2], dim=0)
    combined_faces = torch.cat([
        faces1, 
        faces2 + verts1.shape[0]  # 偏移面索引
    ], dim=0)

    return Meshes(verts=[combined_verts], faces=[combined_faces])


def test(dir):
    paths = ["6004", "7466", "7861", "7885", "0126", "8022"]
    for p in paths:
        print(p)
        pa = os.path.join(dir, '0000'+p+'.ply')
        wrapper = TrimeshWrapper.load(pa)
        wrapper.fix_face_orientation()

        wrapper.save(path=f'/home/lkh/siga/output/temp/nor/{p}.ply')


def main():
    dir = '/home/lkh/siga/dataset/my_dataset/cad_ply/addbody/result'
    test(dir)


if __name__ == "__main__":
    main()