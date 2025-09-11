import argparse, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import fields

from utils import load_json_file


class BaseConfig:
    
    @classmethod
    def get_default(cls, config_path='/home/lkh/siga/CADIMG/config/train_config.json'):
        return cls(**load_json_file(config_path))

    @classmethod
    def from_cli(cls, config_path):
        defaults = cls.get_default(config_path)
        parser = argparse.ArgumentParser()

        for field in fields(cls):
            name = field.name
            default_val = getattr(defaults, name)
            field_type = field.type

            if field_type == bool:
                parser.add_argument(f"--{name}", action="store_true", default=default_val)
            elif field_type == list:
                parser.add_argument(f"--{name}", type=int, nargs='+', default=default_val)
            else:
                parser.add_argument(f"--{name}", type=field_type, default=default_val)

        args = parser.parse_args()
        return cls(**vars(args))