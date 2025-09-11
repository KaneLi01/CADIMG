import os
import utils.cadlib.Brep_utils as Brep_utils
import utils.vis.render_cad as render_cad
from dataset.process.ABC.childnum1_process import Childnum1BaseAddMerge
from rewrite_unvalid_add_shape import rescale_sphere

def main(name):
 # 00817241 00602007
    root_dir = '/home/lkh/siga/dataset/ABC/childnum1_base_add_shape/1/step'
    base_path = os.path.join(root_dir, 'base', name + '.step')
    add_path = os.path.join(root_dir, 'add', name + '.step')

    base_shape = Brep_utils.get_BRep_from_step(base_path)
    #add_shape = Brep_utils.normalize_shape(Brep_utils.get_BRep_from_step(add_path))
    add_shape = Brep_utils.get_BRep_from_step(add_path)

    c1bam = Childnum1BaseAddMerge()
    add_shape_srt, view = c1bam.compute_trans_add_shape_view(base_shape, add_shape, scale_factor=1.0)

    add_shape_new = rescale_sphere(base_shape, add_shape)

    render_cad.display_BRep_list_with_different_color([base_shape, add_shape], ['blue', 'green'])