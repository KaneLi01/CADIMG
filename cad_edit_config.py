from dataclasses import dataclass
import argparse, os
from typing import Optional
from utils import log_util

DEFAULT_CONFIG = {
    # 预训练模型路径
    "sd_path": "/home/lkh/siga/ckpt/sd15",
    "controlnet_path": "/home/lkh/siga/ckpt/controlnet_scribble",
    "img_encoder_path": "/home/lkh/siga/ckpt/clip-vit-large-patch14",
    "projector_path": "/home/lkh/siga/ckpt/projector_weights.pth",
    
    # 训练controlnet路径名称
    "parent_cn_path": "/home/lkh/siga/CADIMG/log",
    "index": "0508_1848",
    
    # 测试图像
    "test_img_dir": "/home/lkh/siga/dataset/deepcad/data/cad_controlnet01/init_img/",
    "test_sketch_dir": "/home/lkh/siga/dataset/deepcad/data/cad_controlnet01/stroke_img/",
    "img_index": [0,4],
    
    # 输出路径
    "output_dir": "/home/lkh/siga/CADIMG/infer",
    
    # 其他
    "device": "cpu",
    "tip": " "
}


@dataclass
class AppConfig:
    """预训练模型路径"""
    sd_path: str
    controlnet_path: str
    img_encoder_path: str
    projector_path: str

    """训练controlnet路径名称"""
    parent_cn_path: str
    index: str
    
    """测试图像"""
    test_img_dir: str
    test_sketch_dir: str
    img_index: list

    """输出路径"""
    output_dir: str

    """备注"""
    device: str
    tip: str
    

    @classmethod
    def get_default(cls):
        return cls(**DEFAULT_CONFIG)

    @classmethod
    def from_cli(cls):
        """Create config from command line arguments"""
        defaults = cls.get_default()
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--sd_path", 
                          default=defaults.sd_path)
        parser.add_argument("--controlnet_path", 
                          default=defaults.controlnet_path)
        parser.add_argument("--img_encoder_path", 
                          default=defaults.img_encoder_path)
        parser.add_argument("--projector_path", 
                          default=defaults.projector_path)
        
        parser.add_argument("--parent_cn_path", 
                          default=defaults.parent_cn_path)
        parser.add_argument("--index",
                          default=defaults.index)
        
        parser.add_argument("--test_img_dir", 
                          default=defaults.test_img_dir)
        parser.add_argument("--test_sketch_dir", 
                          default=defaults.test_sketch_dir)
        parser.add_argument("--img_index", type=int, nargs='+', 
                          default=defaults.img_index)
        
        parser.add_argument("--output_dir", 
                          default=defaults.output_dir)
        
        parser.add_argument("--device", 
                          default=defaults.device)
        parser.add_argument("--tip", 
                          default=defaults.tip)
        
        args = parser.parse_args()
        config = cls(
            sd_path=args.sd_path,
            controlnet_path=args.controlnet_path,
            img_encoder_path=args.img_encoder_path,
            projector_path=args.projector_path,

            parent_cn_path=args.parent_cn_path,
            index=args.index,
            
            test_img_dir=args.test_img_dir,
            test_sketch_dir=args.test_sketch_dir,
            img_index=args.img_index,

            output_dir=args.output_dir,

            device=args.device,
            tip=args.tip,
        )

        return config


        

        
