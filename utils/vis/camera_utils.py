import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import *
import cadlib.Brep_utils as Brep_utils
from . import render_cad
import trimesh

def get_view(base_min, base_max, op_min, op_max, thre=1):
    '''利用包围盒计算观测角度，以让add的sketch不会被mask'''
    if type(base_min) is list:
        base_min = Brep_utils.Point.from_list(base_min)
        base_max = Brep_utils.Point.from_list(base_max)
        op_min = Brep_utils.Point.from_list(op_min)
        op_max = Brep_utils.Point.from_list(op_max)


    if base_max.z - op_min.z <= thre:
        view = 'up' 
    elif base_min.z - op_max.z >= -thre:
        view = 'down'
    elif base_max.x - op_min.x <= thre:
        view = 'front'
    elif base_min.x - op_max.x >= -thre:
        view = 'back'
    elif base_max.y - op_min.y <= thre:
        view = 'right'
    elif base_min.y - op_max.y >= -thre:
        view = 'left'
    else: view = None

    return view


def get_view_axis(base_center, op_center, thre=0):
    '''利用包围盒中心计算观测角度，以让add的sketch不会被mask'''
    dx = op_center.x - base_center.x
    dy = op_center.y - base_center.y
    dz = op_center.z - base_center.z

    m = max(dx, dy, dz, key=abs)

    if abs(m) == abs(dx):
        if dx > thre:
            view = 'front'
        else:
            view = 'back'
    elif abs(m) == abs(dy):
        if dy > thre:
            view = 'right'
        else:
            view = 'left'
    elif abs(m) == abs(dz):
        if dz > thre:
            view = 'up'
        else:
            view = 'down'
    else:
        view = None
        raise Exception('view compute: something wrong')
    
    return view


def get_opposite_view(view):
    opposite_map = {
        'up': 'down',
        'down': 'up',
        'front': 'back',
        'back': 'front',
        'right': 'left',
        'left': 'right'
    }
    return opposite_map.get(view, view)


def compute_cam_pos_8(see_at=(0.0, 0.0, 0.0), bbox_size=0.75, view='up', scale=2):
    from math import sqrt
    
    # 固定中心点为 (0, 0, 0)
    x, y, z = see_at[0], see_at[1], see_at[2]
    
    max_length = bbox_size  # 最大边长
    offset = max_length * scale / 2
    diag = offset * sqrt(2) / 2

    if view == 'up':
        z_cam = z + offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - diag, y + diag, z_cam],
            [x - offset, y, z_cam],
            [x - diag, y - diag, z_cam]
        ]
    elif view == 'down':
        z_cam = z - offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - diag, y + diag, z_cam],
            [x - offset, y, z_cam],
            [x - diag, y - diag, z_cam]
        ]
    elif view == 'front':
        x_cam = x + offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y + diag, z - diag],
            [x_cam, y, z - offset],
            [x_cam, y - diag, z - diag]
        ]
    elif view == 'back':
        x_cam = x - offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y + diag, z - diag],
            [x_cam, y, z - offset],
            [x_cam, y - diag, z - diag]
        ]
    elif view == 'right':
        y_cam = y + offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x + diag, y_cam, z - diag],
            [x, y_cam, z - offset],
            [x - diag, y_cam, z - diag]
        ]
    elif view == 'left':
        y_cam = y - offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x + diag, y_cam, z - diag],
            [x, y_cam, z - offset],
            [x - diag, y_cam, z - diag]
        ]
    else:
        raise ValueError("Unknown view name: " + view)

    return cam_pos_list


def compute_cam_pos_mesh(mesh, view='up', scale=1.2):
    # bbox = mesh.bounding_box
    # bbox_center = bbox.center_mass  # 中心点 (x, y, z) 的 numpy 数组
    # x, y, z = bbox_center[0], bbox_center[1], bbox_center[2]

    # # 2. 计算 offset 和 diag
    # bbox_extents = bbox.extents  # 包围盒尺寸 (width, height, depth)
    # max_length = max(bbox_extents)  # 替换原 bbox.max_length()
    # offset = max_length * scale / 2
    # diag = offset * sqrt(2) / 2

    # 获取包围盒的顶点坐标
    verts = mesh.verts_packed()  # [V, 3]
    
    # 计算包围盒的中心点 (x, y, z)
    bbox_center = verts.mean(dim=0)  # [3]
    x, y, z = bbox_center[0], bbox_center[1], bbox_center[2]
    
    # 计算包围盒的尺寸 (width, height, depth)
    min_vals = verts.min(dim=0)[0]  # [3]
    max_vals = verts.max(dim=0)[0]  # [3]
    bbox_extents = max_vals - min_vals  # [3]
    
    max_length = bbox_extents.max().item()  # 最大边长
    offset = max_length * scale / 2
    diag = offset * sqrt(2) / 2


    if view == 'up':
        z_cam = z + offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - offset, y, z_cam]
        ]
    elif view == 'down':
        z_cam = z - offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - offset, y, z_cam]
        ]
    elif view == 'front':
        x_cam = x + offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y, z - offset]
        ]
    elif view == 'back':
        x_cam = x - offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y, z - offset]
        ]
    elif view == 'right':
        y_cam = y + offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x, y_cam, z - offset]
        ]
    elif view == 'left':
        y_cam = y - offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x, y_cam, z - offset]
        ]
    else:
        raise ValueError("Unknown view name: " + view)

    return cam_pos_list  


def compute_cam_pos_mesh_8directions(mesh, view='up', scale=1.2):
    # 获取包围盒的顶点坐标
    verts = mesh.verts_packed()  # [V, 3]
    
    # 计算包围盒的中心点 (x, y, z)
    bbox_center = verts.mean(dim=0)  # [3]
    x, y, z = bbox_center[0], bbox_center[1], bbox_center[2]
    
    # 计算包围盒的尺寸 (width, height, depth)
    min_vals = verts.min(dim=0)[0]  # [3]
    max_vals = verts.max(dim=0)[0]  # [3]
    bbox_extents = max_vals - min_vals  # [3]
    
    max_length = bbox_extents.max().item()  # 最大边长
    offset = max_length * scale / 2
    diag = offset * sqrt(2) / 2


    if view == 'up':
        z_cam = z + offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - diag, y + diag, z_cam],
            [x - offset, y, z_cam],
            [x - diag, y - diag, z_cam]
        ]
    elif view == 'down':
        z_cam = z - offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - diag, y + diag, z_cam],
            [x - offset, y, z_cam],
            [x - diag, y - diag, z_cam]
        ]
    elif view == 'front':
        x_cam = x + offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y + diag, z - diag],
            [x_cam, y, z - offset],
            [x_cam, y - diag, z - diag]
        ]
    elif view == 'back':
        x_cam = x - offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y + diag, z - diag],
            [x_cam, y, z - offset],
            [x_cam, y - diag, z - diag]
        ]
    elif view == 'right':
        y_cam = y + offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x + diag, y_cam, z - diag],
            [x, y_cam, z - offset],
            [x - diag, y_cam, z - diag]
        ]
    elif view == 'left':
        y_cam = y - offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x + diag, y_cam, z - diag],
            [x, y_cam, z - offset],
            [x - diag, y_cam, z - diag]
        ]
    else:
        raise ValueError("Unknown view name: " + view)

    return cam_pos_list  


def compute_cam_pos_step(shape, view='up', scale=1.2):
    bbox = Brep_utils.get_bbox(shape)
    center = bbox.center
    x, y, z = center.x, center.y, center.z
    offset = bbox.max_length() * scale / 2
    diag = offset * sqrt(2) / 2

    if view == 'up':
        z_cam = z + offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - offset, y, z_cam]
        ]
    elif view == 'down':
        z_cam = z - offset
        cam_pos_list = [
            [x, y - offset, z_cam],
            [x + diag, y - diag, z_cam],
            [x + offset, y, z_cam],
            [x + diag, y + diag, z_cam],
            [x, y + offset, z_cam],
            [x - offset, y, z_cam]
        ]
    elif view == 'front':
        x_cam = x + offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y, z - offset]
        ]
    elif view == 'back':
        x_cam = x - offset
        cam_pos_list = [
            [x_cam, y - offset, z],
            [x_cam, y - diag, z + diag],
            [x_cam, y, z + offset],
            [x_cam, y + diag, z + diag],
            [x_cam, y + offset, z],
            [x_cam, y, z - offset]
        ]
    elif view == 'right':
        y_cam = y + offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x, y_cam, z - offset]
        ]
    elif view == 'left':
        y_cam = y - offset
        cam_pos_list = [
            [x - offset, y_cam, z],
            [x - diag, y_cam, z + diag],
            [x, y_cam, z + offset],
            [x + diag, y_cam, z + diag],
            [x + offset, y_cam, z],
            [x, y_cam, z - offset]
        ]
    else:
        raise ValueError("Unknown view name: " + view)

    return cam_pos_list  


def test_cam_pos():
    import copy
    step_path = '/home/lkh/siga/dataset/ABC/temp/exam/68/00680000/00680000_58e44015b4d9da0f6f9fdbf2_step_009.step'
    shape = Brep_utils.get_BRep_from_step(step_path)
    shape1, shape2 = copy.deepcopy(shape), copy.deepcopy(shape)
    bbox = Brep_utils.get_bbox(shape1)
    center = bbox.center.to_list()
    cam_poss = compute_cam_pos(shape1, view='down')
    for i in range(6):
        render_cad.save_BRep(output_path=f'/home/lkh/siga/output/temp/t{i}.png',shape=shape1,cam_pos=cam_poss[i],see_at=center)
    render_cad.save_BRep(output_path=f'/home/lkh/siga/output/temp/1.png',shape=shape1,cam_pos=[75,75,75],see_at=[0,0,0])
    render_cad.display_BRep(shape)

    # shape3 = sf(shape2)
    # render_cad.save_BRep(output_path='/home/lkh/siga/output/temp/3.png',shape=shape1,cam_pos=[75,75,75],see_at=[-25,0,-36])
    # render_cad.save_BRep(output_path='/home/lkh/siga/output/temp/4.png',shape=shape3)
    # compute_cam_pos()


def test():

    test_cam_pos()


if __name__ == "__main__":
    test()