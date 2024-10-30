import wandb
from ultralytics import YOLO
import argparse

def train_yolov10(weight_path, wandb_key, data_yaml):
    wandb.login(key=wandb_key)

    model = YOLO(weight_path)
    results = model.train(data=data_yaml, epochs=15, imgsz=832, close_mosaic=15, single_cls=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train yolov10")
    parser.add_argument(
        "-w",
        "--wandb_key",
        type=str,
    )
    parser.add_argument(
        "--yaml",
        type=str,
        help="path to yaml file",
    )
    parser.add_argument(
        "--weigth_path",
        type=str,
        default="yolov10l.pt"
    )

    args = parser.parse_args()
    train_yolov10(args.weight_path, args.wandb_key, args.yaml)