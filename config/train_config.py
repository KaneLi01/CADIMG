import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from .base_config import BaseConfig
from utils import log_string


@dataclass
class AppConfig(BaseConfig):
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
    mode: str
    device: str
    res: int
    num_epochs: int
    batch_size: int
    lam: float
    lr: float
    torch_dtype: str
    weight_decay: float

    """备注"""
    debug: bool
    tip: str
    

    @classmethod
    def write_config(cls, config_obj, log_file=None, compare_log=None):
        default_config = cls.get_default()
        if log_file:
            for field in config_obj.__dataclass_fields__:
                value = getattr(config_obj, field)
                if log_file:
                    log_string(f"{field}: {value}", log_file)

                # 如果当前值与默认值不同，写入 compare_log
                if compare_log and value != getattr(default_config, field):
                    compare_log.write(f"{field}: {value}\t")
                    compare_log.flush()
        

        
