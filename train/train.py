"""
Training — finetune YOLOv11 on damage dataset

Usage: python train.py --data dataset.yaml --model yolo11n.pt --epochs 100
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch: int = 8,
    lr: float = 0.01,
    patience: int = 20,
    imgsz: int = 640,
    augment_mosaic: float = 0.5,
    augment_hsv_h: float = 0.015,
    augment_hsv_s: float = 0.7,
    augment_hsv_v: float = 0.4,
    augment_degrees: float = 10.0,
    augment_scale: float = 0.5,
    device: str = "cpu",
    project: str = "models",
    name: str = "damage_detector",
    workers: int = 8,
    cache: bool = True,
    amp: bool = True,
):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr,
        patience=patience,
        imgsz=imgsz,
        augment=True,
        mosaic=augment_mosaic,
        hsv_h=augment_hsv_h,
        hsv_s=augment_hsv_s,
        hsv_v=augment_hsv_v,
        degrees=augment_degrees,
        scale=augment_scale,
        device=device,
        workers=workers,
        cache=cache,
        amp=amp,
        project=project,
        name=name,
        exist_ok=True,
    )
    return Path(project) / name / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset.yaml path")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    best = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
    )
    print(f"Training complete. Best model: {best}")


if __name__ == "__main__":
    main()
