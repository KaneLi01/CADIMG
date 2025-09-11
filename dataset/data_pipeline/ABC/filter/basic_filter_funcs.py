from itertools import combinations
from dataset.prepare_data.ABC.shape_info import Get_Feat_From_Dict

class Filter_Name_From_Feat(Get_Feat_From_Dict): 
    def filter_null(self):
        if not self.valid:
            return False
        else: return True

    def filter_child_num(self):
        if self.child_num > 20:
            return False
        else: return True

    def filter_complex_faces(self, thre=20):
        '''筛选所有shape面数是否小于阈值'''
        for shape in self.sub_shapes:
            num_face = shape.face_num
            if num_face >= thre:
                return False
        return True

    def filter_complex_wires(self, thre=64):
        '''筛选所有shape线数是否小于阈值'''
        for shape in self.sub_shapes:
            num_wire = shape.wire_num
            if num_wire >= thre:
                return False
        return True

    def filter_simple_faces(self, thre=10):
        '''筛选至少一个面数小于阈值'''
        if self.child_num == 2:
            for shape in self.sub_shapes:
                num_face = shape.face_num
                if num_face <= thre:
                    return True
            return False 
        if 3 <= self.child_num <=20:
            n1 = 0
            for shape in self.sub_shapes[:-1]:
                num_face = shape.face_num
                n1 += num_face
            n2 = self.sub_shapes[-1].face_num
            if n1 <= thre or n2 <= thre:
                return True
            else: return False

    def filter_small_thin(self, scale=10):
        '''筛选所有shape是否不是薄面或棍'''
        def check_slice(dx, dy, dz, scale):
            '''如果任意两边长度比大于scale，则true'''
            for a, b in combinations([dx, dy, dz], 2):
                longer = max(a, b)
                shorter = min(a, b)
                if longer >= scale * shorter:
                    return True
            return False
        
        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]     
            dx, dy, dz = bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]

            if check_slice(dx, dy, dz, scale):
                return False
        return True

    def filter_stick(self, scale=10):
        '''筛选所有 shape 是否不是棍状'''
        def is_stick(dx, dy, dz, scale):
            """最长边如果是其他两个边的 scale 倍及以上，就判定为棍"""
            edges = [dx, dy, dz]
            longest = max(edges)
            edges.remove(longest)  # 剩下两个短边
            return all(longest >= scale * e for e in edges)

        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]
            dx, dy, dz = (
                bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2],
            )

            if is_stick(dx, dy, dz, scale):
                return False  # 棍状 → False
        return True  # 全部不是棍状 → True    

    def filter_zero_face(self):
        '''筛选掉child不是shape的情况'''
        if self.child_num == 2:
            if self.sub_shapes[0].face_num == 0 or self.sub_shapes[1].face_num == 0:
                return False
            else: return True    
            
        if 3 <= self.child_num <=20:
            f1 = 0
            for sub_shape in self.sub_shapes:
                f = sub_shape.face_num
                f1 += f
            f2 = self.sub_shapes[-1].face_num

            if f1 == 0 or f2 == 0:
                return False 
            else: return True 

    def filter_intersection(self, thre=1):
        '''筛选两个body是否不相交'''
        if self.common_volume >= thre or self.common_volume <= -thre:
            return False
        else: return True

    def filter_distance(self, thre=1):
        '''筛选两个body最小距离是否极小'''
        if self.min_dis >= thre or self.min_dis <= -thre:
            return False
        else: return True

    def filter_simple_face_wire(self):
        '''筛选掉face为0, 线条数过少的情况'''
        for shape in self.sub_shapes:
            if (shape.face_num == 0) or (shape.wire_num < 4):
                return False
        return True

    def filter_shape_bbox_volume_ratio(self, ratio=4):
        '''筛选掉所有子shape，和其包围盒体积比值过小的情况。可以改善如桶状的对象'''
        for sv, bv in zip(self.shape_volume, self.bbox_volume):
            if sv*ratio < bv:
                return False
        return True

    def filter_solid(self):
        '''筛选所有子形状都为solid'''
        for sva in self.solid_valid:
            if sva == 0:
                return False
        return True

    def filter_screw(self, thre=15, cond=False):
        '''筛选螺丝类型的形状'''  
        if (any(x <= 2 for x in self.face_num)):
            return True  
        if (any(2 < x <= 4 for x in self.face_num) and ((3 in self.faces_type) or (4 in self.faces_type))):
            return True
        for edges_len, face_type in zip(self.edges_len, self.faces_type):
            if cond:
                if face_type == 1 or face_type == 6 or face_type == 7:
                    longest = max(edges_len)
                    shortest = min(edges_len)
                    if shortest*thre < longest:
                        return False
            else:
                longest = max(edges_len)
                shortest = min(edges_len)
                if shortest*thre < longest:
                    return False                
        return True

    def filter_edge_len(self, thre=1/8):
        '''归一化后最短的edge长度'''
        transposed = list(zip(*self.bbox_min_max)) 
        result = []
        for i in range(3):
            result.append(min(transposed[i]))
        # 后三个位置取最大值
        for i in range(3, 6):
            result.append(max(transposed[i]))

        dx, dy, dz = result[3] - result[0], result[4] - result[1], result[5] - result[2]
        longest_bbox_edge = max(dx, dy, dz)

        scale = longest_bbox_edge / 1.5

        for edges_len in self.edges_len:
            for edge_len in edges_len:
                if edge_len / (scale + 0.000000001) < thre:
                    return False
        return True

    def filter_face_area_0(self):
        for face_area in self.faces_area:
            if face_area <= 0.0:
                return False
        return True

    def filter_face_area(self, thre=1/4):
        '''归一化后有plane，且至少有一个plane的面积大于thre'''
        transposed = list(zip(*self.bbox_min_max)) 
        result = []
        for i in range(3):
            result.append(min(transposed[i]))
        for i in range(3, 6):
            result.append(max(transposed[i]))

        dx, dy, dz = result[3] - result[0], result[4] - result[1], result[5] - result[2]
        bbox_face_area = [dx*dy, dx*dz, dy*dz]
        longest_bbox_face_area = max(bbox_face_area)

        for face_area, face_type in zip(self.faces_area, self.faces_type):
            if face_type == 0:
                if face_area / longest_bbox_face_area > thre:
                    return True
        return False

    def classify_plane(self, thre=10):
        min_bbox_face = []
        for shape in self.sub_shapes:
            bbox_min = shape.bbox_min_max[:3]
            bbox_max = shape.bbox_min_max[3:]     
            dx, dy, dz = bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]     
            min_bbox_face.append(min(dx*dy, dx*dz, dy*dz))  
        minimal_bbox_face = min(min_bbox_face)

        for area, type in zip(self.faces_area, self.faces_type):
            if type == 0:
                if area*thre > minimal_bbox_face:
                    return True
                
        return False