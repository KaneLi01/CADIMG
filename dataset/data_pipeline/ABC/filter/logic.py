from basic_filter_funcs import Filter_Name_From_Feat
import utils.cadlib.Brep_utils as Brep_utils
from dataset.data_pipeline.ABC.prepare.extract_feature_json import Shape_Info_From_Step

def distinguish_add_base_1(data):

    feat = Filter_Name_From_Feat(data)

    if (feat.filter_complex_faces(thre=10) 
        and feat.filter_complex_wires(thre=32) 
        and feat.filter_small_thin(scale=3) 
        and feat.filter_screw(thre=20) 
        and feat.filter_edge_len(thre=1/5)
        ):
        return 'add'
    elif feat.filter_complex_faces(thre=4):
        return 'add'
    else:
        return 'base' 


def distinguish_add_base_2(data):
    feat = Filter_Name_From_Feat(data)

    if (feat.filter_complex_faces(thre=15) 
        and feat.filter_complex_wires(thre=48) 
        and feat.filter_small_thin(scale=4) 
        and feat.filter_screw(thre=30) 
        and feat.filter_edge_len(thre=1/8)
        ):
        return 'add'
    elif feat.filter_complex_faces(thre=4):
        return 'add'
    else:
        return 'base' 


def filter_base_valid_plane_1(data):

    feat = Filter_Name_From_Feat(data)
    print(feat.name)
    step_path = Brep_utils.get_step_path_from_name(feat.name)
    shape = Brep_utils.get_BRep_from_step(step_path)
    if feat.filter_face_area(thre=1/5) and (Brep_utils.get_valid_biggest_plane(shape, thre=1/4)[0] is not None):
        return 'valid'
    else:
        return 'unvalid'        


def filter_face_area_0_1(data):
    feat = Filter_Name_From_Feat(data)
    if feat.filter_face_area_0():
        return 'filtered'
    else:
        return 'removed'
    

def distinguish_add_complexity_all_1(data):
    '''plane, cylinder, cone, sphere, torus, beziersurface, bsplinesurface, surfaceofrevolution, surfaceofextrusion, othersurface'''
    feat = Filter_Name_From_Feat(data)
    if feat.face_num[0] <= 2:
        if 4 in feat.faces_type:
            return 'torus'
        elif 2 in feat.faces_type:
            return 'cone'
        elif 3 in feat.faces_type:
            return 'sphere'
        else:
            return '12other'
    else:
        if feat.face_num[0] == 3:
            return '3'
        elif feat.face_num[0] == 4:
            return '4'
        elif feat.face_num[0] == 5:
            return '5'
        elif feat.face_num[0] == 6:
            if abs(1 - feat.bbox_volume[0] / feat.shape_volume[0]) < 0.001:
                return '6cube'
            else:
                return '6other'
        elif feat.face_num[0] == 7:
            return '7'
        elif feat.face_num[0] == 8:
            return '8'
        elif feat.face_num[0] >= 9:
            return '9up'
        else:
            raise Exception('wrong classify')