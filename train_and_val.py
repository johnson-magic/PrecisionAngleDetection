import torch
import random
import numpy as np

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set your desired seed value here

from ultralytics import YOLO

from utils.config_utils import arg_parser, parse_config

def train_and_val(config_file):
    config = parse_config(config_file)
    # Load a model
    model = YOLO(config["model"]["model"]).load(config["model"]["pretrain_model"])
    config["hyperparameter"].update(config["dataset"])
    results = model.train(**config["hyperparameter"])



    
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    train_and_val(args.config)

# e.g.
"""python3 train_and_val --config ./config/config.yaml"""

"""/home/ecs-user/.local/lib/python3.10/site-packages/ultralytics/models/yolo/model.py"""
