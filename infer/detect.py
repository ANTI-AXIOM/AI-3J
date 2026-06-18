"""
Inference — run YOLO on keyframes → per-frame detection JSON
"""

import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO


def infer_keyframes(
    model_path: str,
    keyframe_dir: str,
    output_json: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
) -> list[dict]:
    """
    Run YOLOv11 on all keyframes in a directory.

    Returns: list of {frame_file, detections: [{class_id, class_name, bbox, conf}, ...]}
    """
    model = YOLO(model_path)
    keyframe_dir = Path(keyframe_dir)
    image_paths = sorted(keyframe_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files in {keyframe_dir}")

    results_list = model(
        [str(p) for p in image_paths],
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    per_frame = []
    for img_path, r in zip(image_paths, results_list):
        detections = []
        if r.boxes is not None:
            for box, cls_id, conf_val in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = box.tolist()
                detections.append({
                    "class_id": int(cls_id.item()),
                    "class_name": r.names[int(cls_id.item())],
                    "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                    "confidence": round(conf_val.item(), 3),
                })
        per_frame.append({
            "frame_file": img_path.name,
            "frame_path": str(img_path.resolve()),
            "detections": detections,
        })

    with open(output_json, "w") as f:
        json.dump(per_frame, f, indent=2)

    return per_frame


def infer_single_frame(model, frame_path: str, conf: float = 0.25, iou: float = 0.45):
    """Convenience for single-frame inference (used in pipeline)."""
    results = model(frame_path, conf=conf, iou=iou, verbose=False)
    r = results[0]
    detections = []
    if r.boxes is not None:
        for box, cls_id, conf_val in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "class_id": int(cls_id.item()),
                "class_name": r.names[int(cls_id.item())],
                "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                "confidence": round(conf_val.item(), 3),
            })
    return detections


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python infer.py <model.pt> <keyframes_dir> <output.json>")
        sys.exit(1)
    results = infer_keyframes(sys.argv[1], sys.argv[2], sys.argv[3])
    total_dets = sum(len(f["detections"]) for f in results)
    print(f"Inferred {len(results)} frames, {total_dets} total detections")
