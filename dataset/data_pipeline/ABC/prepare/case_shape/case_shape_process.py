import yaml, os

import utils.cadlib.Brep_utils as Brep_utils
import utils.vis.render_cad as render_cad
import utils.path_file_utils as path_file_utils


'''all_class = ['chess', 'blocks', 'mechanial_part', 'cube', 'window', 'ring', 'bottle', 'pump', 'bearing', 'crankshaft_adapter', 'Tetris']'''
class CaseShapes:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.all_cases = self._read_cases()

    def _read_cases(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        return data
    
    def write_case_to_txt(self, txt_path):
        for class_name in self.all_cases.keys():
            for name in self.all_cases[class_name]:
                path_file_utils.append_line_to_file(txt_path, name)

    def init_render(self, img_output_root_dir=None):
        if img_output_root_dir is None:
            raise Exception('please input the img output root directory')
        
        for class_name in self.all_cases.keys():
            img_output_dir = os.path.join(img_output_root_dir, class_name)
            os.makedirs(img_output_dir, exist_ok=True)

            for name in self.all_cases[class_name]:
                img_output_path = os.path.join(img_output_dir, name + '.png')
                step_path = Brep_utils.get_step_path_from_name(name=name)
                shape = Brep_utils.get_BRep_from_step(step_path)
                render_cad.save_BRep(output_path=img_output_path, shape=shape)



def create_list_name():
    yaml_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/case_shape/case_name.yaml'
    txt_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/case_shape/src/cases.txt'

    cs = CaseShapes(yaml_path)
    cs.write_case_to_txt(txt_path)


def render():
    yaml_path = '/home/lkh/siga/CADIMG/dataset/process/ABC/case_shape/case_name.yaml'
    img_output_root_dir = '/home/lkh/siga/CADIMG/dataset/process/ABC/case_shape/src/vis'

    cs = CaseShapes(yaml_path)
    cs.write_case_to_txt()
    cs.init_render(img_output_root_dir=img_output_root_dir)


def main():
    create_list_name()


if __name__ == '__main__':
    main()