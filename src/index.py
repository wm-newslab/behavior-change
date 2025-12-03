import argparse
import yaml
from pathlib import Path

from .fox8_analyzer import main as fox8_main
from .infoOps_analyzer import main as infoOps_main
from .retraining_analyzer import main as retraining_main
from .user_analyzer import main as user_analyzer_main

def load_config(path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "fox8_analyzer",
            "infoOps_analyzer",
            "retraining_analyzer",
            "user_analyzer",
        ],
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml"
    )
    args = parser.parse_args()

    cfg  =  load_config(args.config)

    if args.task == "fox8_analyzer":
        fox8_main(cfg)

    elif args.task == "infoOps_analyzer":
        infoOps_main(cfg)

    elif args.task == "retraining_analyzer":
        retraining_main(cfg)

    elif args.task == "user_analyzer":
        user_analyzer_main(cfg)

if __name__ == "__main__":
    main()
