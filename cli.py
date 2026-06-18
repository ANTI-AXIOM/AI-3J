"""
Main CLI — run the full pipeline end-to-end
"""

import argparse
import json
import sys
from pathlib import Path

from ingest.extract_keyframes import extract_keyframes
from infer.detect import infer_keyframes
from infer.temporal_agg import run_aggregation
from infer.problem_classifier import run_classifier, load_config


def run_infer(args):
    print(f"[1/4] Extracting keyframes from {args.video}...")
    frames = extract_keyframes(
        video_path=args.video,
        output_dir=args.keyframes_dir,
        scene_threshold=args.config.get("extraction", {}).get("scene_threshold", 30.0),
        blur_variance_min=args.config.get("extraction", {}).get("blur_variance_min", 100.0),
        dedup_corr_threshold=args.config.get("extraction", {}).get("dedup_corr_threshold", 0.95),
        max_frames=args.config.get("extraction", {}).get("max_keyframes_per_video", 50),
    )
    print(f"  → {len(frames)} keyframes")

    if not frames:
        print("  ✗ No valid keyframes extracted (video blurry/empty/too short)")
        return

    detections_json = Path(args.output_dir) / "01_detections.json"
    print(f"[2/4] Running YOLO inference on keyframes...")
    per_frame = infer_keyframes(
        model_path=args.model,
        keyframe_dir=args.keyframes_dir,
        output_json=str(detections_json),
        conf=args.config.get("model", {}).get("conf_threshold", 0.25),
        iou=args.config.get("model", {}).get("iou_threshold", 0.45),
        device=args.config.get("model", {}).get("device", "cpu"),
    )
    total = sum(len(f["detections"]) for f in per_frame)
    print(f"  → {total} detections across {len(per_frame)} frames")

    tracks_json = Path(args.output_dir) / "02_tracks.json"
    print(f"[3/4] Aggregating temporal tracks...")
    tracks = run_aggregation(
        detections_json=str(detections_json),
        output_json=str(tracks_json),
        iou_threshold=args.config.get("temporal_agg", {}).get("iou_threshold", 0.3),
        min_track_frames=args.config.get("temporal_agg", {}).get("min_track_frames", 2),
    )
    print(f"  → {len(tracks)} damage tracks")

    recap_json = Path(args.output_dir) / "03_recap.json"
    print(f"[4/4] Classifying problem + generating recap...")
    recap = run_classifier(
        tracks_json=str(tracks_json),
        output_json=str(recap_json),
        config_path=args.config_path,
    )

    print("\n" + "=" * 60)
    print("RECAP:")
    print(recap["summary"])
    print("=" * 60)
    print(f"\nFull result: {recap_json.resolve()}")


def run_extract(args):
    print(f"Extracting keyframes from {args.video}...")
    frames = extract_keyframes(
        video_path=args.video,
        output_dir=args.output,
        scene_threshold=args.config.get("extraction", {}).get("scene_threshold", 30.0),
        blur_variance_min=args.config.get("extraction", {}).get("blur_variance_min", 50.0),
        dedup_corr_threshold=args.config.get("extraction", {}).get("dedup_corr_threshold", 0.95),
        max_frames=args.config.get("extraction", {}).get("max_keyframes_per_video", 100),
    )
    print(f"✓ Extracted {len(frames)} keyframes to {args.output}")


def run_train(args):
    print(f"Training YOLO with data={args.data} model={args.model} epochs={args.epochs}")
    from train.train import train_yolo
    best = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
        benchmark=args.benchmark,
        freeze=args.freeze,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
    )
    print(f"✓ Best model saved to {best}")


def main():
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(description="Insurance Damage Video Analysis Pipeline")
    parser.add_argument("--config", default="config.yaml", help="config path")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # infer subcommand
    infer_p = subparsers.add_parser("infer", help="Run inference on a video")
    infer_p.add_argument("--video", required=True, help="input video file")
    infer_p.add_argument("--model", default="models/yolo11n.pt",
                         help="YOLO model path or name (default: models/yolo11n.pt)")
    infer_p.add_argument("--keyframes-dir", default="data/keyframes", help="keyframe output dir")
    infer_p.add_argument("--output-dir", default="data/results", help="results output dir")
    infer_p.set_defaults(func=run_infer)

    # train subcommand
    train_p = subparsers.add_parser("train", help="Train YOLO model")

    # extract subcommand
    extract_p = subparsers.add_parser("extract", help="Extract keyframes from video")
    extract_p.add_argument("--video", "-v", required=True, help="input video file")
    extract_p.add_argument("--output", "-o", required=True, help="output directory for keyframes")
    extract_p.set_defaults(func=run_extract)
    train_p.add_argument("--data", required=True, help="dataset.yaml path")
    train_p.add_argument("--model", default="yolo11n.pt", help="base model")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=0.01)
    train_p.add_argument("--workers", type=int, default=8,
                         help="data loader workers")
    train_p.add_argument("--device", default="0",
                         help="device: 0, 1, cpu, or '' for auto-detect")
    train_p.add_argument("--benchmark", default="",
                         help="path to write benchmark JSON (e.g. benchmark.json)")
    train_p.add_argument("--freeze", type=int, default=10,
                         help="freeze first N backbone layers (0=no freeze, default=10)")
    train_p.add_argument("--warmup-epochs", type=int, default=5,
                         help="warmup epochs for LR ramp-up (default=5)")
    train_p.add_argument("--cos-lr", action="store_true", default=True,
                         help="use cosine LR scheduler (default=True)")
    train_p.set_defaults(func=run_train)

    args = parser.parse_args()
    args.config = config
    args.config_path = "config.yaml"

    Path(args.output_dir if hasattr(args, "output_dir") else "data/results").mkdir(
        parents=True, exist_ok=True
    )
    args.func(args)


if __name__ == "__main__":
    main()
