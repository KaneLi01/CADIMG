import utils.cadlib.Brep_utils as Brep_utils

class Shape_Info_From_Step():
    '''从step文件中读取Brep的基础信息'''
    def __init__(self, step_path):
        self.step_path = step_path
        self.cad_name = self.step_path.split('/')[-1].split('_')[0]

        try:
            self.shape = Brep_utils.get_BRep_from_step(self.step_path)
            try:
                self.sub_shapes = Brep_utils.get_child_shapes(self.shape)
            except Exception as e:
                print(f"Error getting sub-shapes from {self.step_path}: {e!r}")
                self.sub_shapes = []
        except Exception as e:
            print(f"Error reading STEP file {self.step_path}: {e!r}")
            self.shape = None
            self.sub_shapes = []

    @property
    def child_num(self):
        return self.shape.NbChildren() if self.shape else 0

    @property
    def edge_num(self):
        return len(Brep_utils.get_edges(self.shape)) if self.shape else 0

    @property
    def face_num(self):
        return len(Brep_utils.get_faces(self.shape)) if self.shape else 0

    @property
    def bbox_feat(self):
        if not self.sub_shapes:
            return []
        bboxs = [Brep_utils.get_bbox(sub_shape) for sub_shape in self.sub_shapes]
        return {
            "min_max_pt": [bbox.min.to_list() + bbox.max.to_list() for bbox in bboxs],
            "center": [bbox.center.to_list() for bbox in bboxs]
        }

    def summary(self):
        """一次性获取所有基础信息"""
        return {
            "cad_name": self.cad_name,
            "child_num": self.child_num,
            "edge_num": self.edge_num,
            "face_num": self.face_num,
            "bbox_feat": self.bbox_feat,
        }