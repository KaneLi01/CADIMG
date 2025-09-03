from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class AppConfig(BaseConfig):
    """预训练模型路径"""
    sd_path: str
    controlnet_path: str
    img_encoder_path: str
    projector_path: str

    """训练controlnet路径名称"""
    parent_cn_path: str
    index: str
    
    """测试图像"""
    test_img_path: str
    test_sketch_path: str
    img_index: list
    res: int

    """输出路径"""
    output_dir: str

    """模型"""
    torch_dtype: str

    """备注"""
    device: str
    tip: str
    