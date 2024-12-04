from ultralytics import YOLO

from utils.config_utils import arg_parser, parse_config

def train_and_val(config_file):
    config = parse_config(config_file)
    # Load a model
    model = YOLO(config["model"]["model"]).load(config["model"]["pretrain_model"])
    results = model.train(**config["hyperparameter"])



    
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    train_and_val(args.config)

# e.g.
"""python3 train_and_val --config ./config/config.yaml"""
