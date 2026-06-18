"""
Temporal Aggregation — IOU-track detections across frames
→ stable damage instances with averaged confidence.
"""

from collections import defaultdict
import json


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IOU of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assign_location(class_name: str, bbox_center_x: float, image_width: int = 640) -> str:
    """Heuristic location name based on class + horizontal position."""
    location_map = {
        "dent": ["hood", "door", "fender", "roof", "trunk"],
        "crack": ["bumper", "windshield", "wall", "window", "ceiling"],
        "scratch": ["door", "bumper", "fender", "wall", "hood"],
        "broken_glass": ["windshield", "window", "headlight"],
        "water_damage": ["ceiling", "wall", "floor"],
        "structural_crack": ["wall", "foundation", "ceiling"],
        "broken_window": ["window", "door_window"],
        "roof_damage": ["roof", "ceiling"],
        "collision_damage": ["bumper", "fender", "hood", "door"],
    }
    zones = location_map.get(class_name, ["body", "wall", "surface"])
    # Use horizontal position: left 1/3, middle 1/3, right 1/3
    rel_x = bbox_center_x / image_width
    idx = 0 if rel_x < 0.33 else (1 if rel_x < 0.66 else 2)
    return zones[min(idx, len(zones) - 1)]


def aggregate_tracks(
    per_frame: list[dict],
    iou_threshold: float = 0.3,
    min_track_frames: int = 2,
    image_width: int = 640,
) -> list[dict]:
    """
    Link detections across frames into tracks.

    Input: per_frame = [{frame_file, detections: [{class_id, class_name, bbox, conf}, ...]}, ...]
    Output: [{track_id, class_name, avg_conf, severity, location, frames: [...]}, ...]
    """
    tracks: list[dict] = []
    active_tracks: list[dict] = []  # tracks from previous frame, awaiting match

    for frame_idx, frame_data in enumerate(per_frame):
        matched = [False] * len(frame_data["detections"])
        unmatched_tracks = list(range(len(active_tracks)))

        for det_idx, det in enumerate(frame_data["detections"]):
            best_iou = 0.0
            best_track = -1
            for t_idx in unmatched_tracks:
                t = active_tracks[t_idx]
                iou = compute_iou(det["bbox"], t["last_bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = t_idx
            if best_iou >= iou_threshold and best_track >= 0:
                t = active_tracks[best_track]
                t["detections"].append(det)
                t["last_bbox"] = det["bbox"]
                t["end_frame"] = frame_idx
                matched[det_idx] = True
                unmatched_tracks.remove(best_track)

        # unmatched active tracks → finalize if old enough
        for t_idx in unmatched_tracks:
            t = active_tracks[t_idx]
            if frame_idx - t["end_frame"] >= 3 or len(t["detections"]) >= min_track_frames:
                tracks.append(_finalize_track(t))
            # keep young unmatched tracks 1 more frame
            else:
                pass  # will be caught in next iteration if still unmatched

        # unmatched detections → new tracks
        for det_idx, det in enumerate(frame_data["detections"]):
            if not matched[det_idx]:
                active_tracks.append({
                    "detections": [det],
                    "last_bbox": det["bbox"],
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                })

    # finalize remaining active tracks
    for t in active_tracks:
        if len(t["detections"]) >= min_track_frames:
            tracks.append(_finalize_track(t))

    return tracks


def _finalize_track(t: dict) -> dict:
    class_counts: dict = {}
    confs: list = []
    all_bboxes: list = []
    for d in t["detections"]:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
        confs.append(d["confidence"])
        all_bboxes.append(d["bbox"])

    class_name = max(class_counts, key=class_counts.get)
    avg_conf = sum(confs) / len(confs)

    # centroid of first detection for location
    first_bbox = t["detections"][0]["bbox"]
    center_x = (first_bbox[0] + first_bbox[2]) / 2
    location = assign_location(class_name, center_x)

    # severity heuristic
    num_classes = len(class_counts)
    if num_classes >= 3:
        severity = "high"
    elif num_classes >= 2:
        severity = "medium"
    else:
        severity = "low" if avg_conf >= 0.6 else "medium"

    return {
        "track_id": id(t),
        "class_name": class_name,
        "avg_confidence": round(avg_conf, 3),
        "severity": severity,
        "location": location,
        "frames": len(t["detections"]),
        "start_frame": t["start_frame"],
        "end_frame": t["end_frame"],
    }


def run_aggregation(
    detections_json: str,
    output_json: str,
    iou_threshold: float = 0.3,
    min_track_frames: int = 2,
) -> list[dict]:
    with open(detections_json) as f:
        per_frame = json.load(f)
    tracks = aggregate_tracks(per_frame, iou_threshold, min_track_frames)
    with open(output_json, "w") as f:
        json.dump(tracks, f, indent=2)
    return tracks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python temporal_agg.py <detections.json> <output.json>")
        sys.exit(1)
    tracks = run_aggregation(sys.argv[1], sys.argv[2])
    print(f"Aggregated {len(tracks)} damage tracks")
    for t in tracks:
        print(f"  #{t['track_id']}: {t['class_name']} @ {t['location']} ({t['severity']}, conf={t['avg_confidence']})")
