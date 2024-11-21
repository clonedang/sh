import wandb
from ultralytics import YOLO
import argparse

def train_yolov11_custom(weight_path, num_epochs, wandb_key, data_yaml):
    wandb.login(key=wandb_key)

    model = YOLO(weight_path)
    results = model.train(data=data_yaml, epochs=num_epochs, imgsz=1280, momentum=0.737, weight_decay=0.0001, warmup_bias_lr=0.01, box=10, cls=2, mosaic=0.5, batch=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train yolov11 custom")
    parser.add_argument(
        "-w",
        "--wandb_key",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
    )
    parser.add_argument(
        "--yaml",
        type=str,
        help="path to yaml file",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="yolov11l.pt"
    )

    args = parser.parse_args()
    train_yolov11_custom(args.weight_path, args.epochs, args.wandb_key, args.yaml)