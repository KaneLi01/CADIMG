
import numpy as np
import trimesh
import pickle
import copy, os, sys
from math import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils.cadlib.Brep_utils as Brep_utils
from CADIMG.utils.vis import render_cad
from shape_info import Shapeinfo, get_args


def main():
    args = get_args()
    shape = Shapeinfo(args.name, args.num)
    output_path = os.path.join('/home/lkh/siga/CADIMG/experiments/test/output/imgs/', args.name+'_sketch.png')
    render_cad.save_BRep_wire_img_temp(shape.wires, campos=shape.campos, seeat=shape.seeat, output_path=output_path)


if __name__ == "__main__":
    main()

