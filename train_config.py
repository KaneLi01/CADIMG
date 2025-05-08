from dataclasses import dataclass
import argparse, os
from typing import Optional
from utils import log_util

DEFAULT_CONFIG = {
    # 模型路径
    "sd_path": "/home/lkh/siga/ckpt/sd15",
    "controlnet_path": "/home/lkh/siga/ckpt/controlnet_scribble",
    "img_encoder_path": "/home/lkh/siga/ckpt/clip-vit-large-patch14",
    "projector_path": "/home/lkh/siga/ckpt/projector_weights.pth",
    
    # 数据路径
    "file_path": "/home/lkh/siga/dataset/deepcad/data/cad_controlnet02",
    
    # 输出配置
    "parent_log_dir": "/home/lkh/siga/CADIMG/log",
    "compare_log": "/home/lkh/siga/CADIMG/compare.log",
    
    # 训练参数
    "device": "cuda:0",
    "num_epochs": 10,
    "batch_size": 4,
    "lam": 0.0,
    
    # 其他
    "tip": " "
}


@dataclass
class AppConfig:
    """模型路径"""
    sd_path: str
    controlnet_path: str
    img_encoder_path: str
    projector_path: str

    """数据集路径"""
    file_path: str
    
    """输出配置"""
    parent_log_dir: str
    compare_log: str

    """训练参数"""
    device: str
    num_epochs: int
    batch_size: int
    lam: float

    """备注"""
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
        
        parser.add_argument("--file_path", 
                          default=defaults.file_path)
        
        parser.add_argument("--parent_log_dir", 
                          default=defaults.parent_log_dir)
        parser.add_argument("--compare_log", 
                          default=defaults.compare_log)
        
        parser.add_argument("--device", 
                          default=defaults.device)
        parser.add_argument("--num_epochs",  type=int,
                          default=defaults.num_epochs)
        parser.add_argument("--batch_size",  type=int,
                          default=defaults.batch_size)
        parser.add_argument("--lam", type=float,
                          default=defaults.lam)
        
        parser.add_argument("--tip", 
                          default=defaults.tip)
        
        args = parser.parse_args()
        config = cls(
            sd_path=args.sd_path,
            controlnet_path=args.controlnet_path,
            img_encoder_path=args.img_encoder_path,
            projector_path=args.projector_path,

            file_path=args.file_path,
            
            parent_log_dir=args.parent_log_dir,
            compare_log=args.compare_log,

            device=args.device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lam=args.lam,

            tip=args.tip,
        )

        return config

    @classmethod
    def write_config(cls, config_obj, log_file=None, compare_log=None):
        if log_file:
            for field in config_obj.__dataclass_fields__:
                log_util.log_string(f"{field}: {getattr(config_obj, field)}", log_file)
                if getattr(config_obj, field) != DEFAULT_CONFIG[field]:
                    compare_log.write(f"{field}: {getattr(config_obj, field)}\t")
                    compare_log.flush()
        

        
