import yaml
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="工业高精度角度检测程序")
    
    # 添加参数
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')

    return parser

def parse_config(config_file: str):
    if config_file.endswith(".yaml"):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config  
    else:
        raise NotImplementedError