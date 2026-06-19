"""
Generate presentation visuals: training curve, dataset distribution, detection examples.

Usage:
    python tools/make_visuals.py

Requires: matplotlib, seaborn (pip install matplotlib seaborn)
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("presentation/imgs")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def plot_training_curve(benchmark_json="benchmark.json"):
    """Plot loss, it/s, and GPU util from benchmark output."""
    if not Path(benchmark_json).exists():
        print(f"  ! {benchmark_json} not found - run training with --benchmark first")
        return

    with open(benchmark_json) as f:
        data = json.load(f)

    if not data:
        print("  ! Empty benchmark data")
        return

    epochs = [e["epoch"] for e in data]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    if "loss" in data[0] and data[0]["loss"]:
        for key in data[0]["loss"]:
            vals = [e["loss"].get(key, 0) for e in data]
            axes[0].plot(epochs, vals, label=key)
        axes[0].set_title("Loss")
        axes[0].legend()

    # it/s
    its = [e.get("it_per_sec", 0) for e in data]
    axes[1].plot(epochs, its, "g-o", markersize=4)
    axes[1].set_title("Throughput (it/s)")

    # GPU util
    gpu = [e.get("gpu_util_pct", 0) for e in data]
    axes[2].plot(epochs, gpu, "r-o", markersize=4)
    axes[2].set_title("GPU Util %")

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=200, bbox_inches="tight")
    print(f"  -> {OUT / 'training_curve.png'}")


def plot_dataset_distribution():
    """Bar chart of samples per class from merged dataset."""
    labels_dir = Path("dataset_frames/labels")
    if not labels_dir.exists():
        print("  ! dataset_frames/labels not found")
        return

    class_counts = {}
    for lbl_file in labels_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    if not class_counts:
        # Fallback: count unique class IDs per file
        for lbl_file in labels_dir.glob("*.txt"):
            classes_in_file = set()
            for line in lbl_file.read_text().strip().split("\n"):
                if line.strip():
                    classes_in_file.add(int(line.split()[0]))
            for c in classes_in_file:
                class_counts[c] = class_counts.get(c, 0) + 1

    if not class_counts:
        print("  ! No labels found")
        return

    ids = sorted(class_counts.keys())
    counts = [class_counts[i] for i in ids]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ids, counts, width=0.8)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Samples")
    ax.set_title(f"Dataset Distribution ({len(ids)} classes, {sum(counts):,} labels)")

    top5 = sorted(zip(ids, counts), key=lambda x: -x[1])[:5]
    for cid, cnt in top5:
        ax.annotate(f"ID {cid}: {cnt}", (cid, cnt), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "dataset_distribution.png", dpi=200, bbox_inches="tight")
    print(f"  -> {OUT / 'dataset_distribution.png'}")


def make_detection_grid(model_path="models/damage_detector/weights/best.pt",
                        sample_images=None):
    """Generate YOLO inference examples on sample images."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ! ultralytics not installed, skipping detection grid")
        return

    if not Path(model_path).exists():
        print(f"  ! Model not found: {model_path}")
        return

    model = YOLO(model_path)

    if sample_images is None:
        import random
        src = Path("dataset_frames/images")
        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        if len(imgs) < 6:
            print("  ! Not enough images for grid")
            return
        sample_images = random.sample(imgs, 6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, img_path in zip(axes.flat, sample_images):
        results = model(str(img_path))
        rendered = results[0].plot()
        ax.imshow(rendered)
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "detection_examples.png", dpi=200, bbox_inches="tight")
    print(f"  -> {OUT / 'detection_examples.png'}")


if __name__ == "__main__":
    print("Generating presentation visuals...")
    plot_training_curve()
    plot_dataset_distribution()
    # make_detection_grid()  # uncomment after training
    print(f"\nAll visuals saved to {OUT.resolve()}")
