"""
KeyFrame Extraction — video → timestamped keyframes
Scene detection + sharpness ranking per scene + histogram dedup.
"""

import cv2
import numpy as np
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def extract_keyframes(
    video_path: str,
    output_dir: str,
    scene_threshold: float = 30.0,
    blur_variance_min: float = 50.0,
    dedup_corr_threshold: float = 0.95,
    max_frames: int = 100,
    verbose: bool = True,
) -> list[dict]:
    """
    Extract representative keyframes from a video.

    Per scene, scans all frames and keeps the sharpest one (highest Laplacian
    variance) that passes the blur threshold.

    Returns: list of {frame_idx, timestamp_sec, path, sharpness_score}
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Opening {video_path.name}...")

    # 1. Scene detection
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if verbose:
        print(f"  Detected {len(scene_list)} scenes")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if not scene_list:
        # fallback: evenly sample up to max_frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            if verbose:
                print(f"  ✗ Cannot open video or 0 frames")
            return []
        n_samples = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
        scene_list = [(int(indices[i]), int(indices[i + 1]))
                      for i in range(len(indices) - 1)]
        if verbose:
            print(f"  Fallback: scanning {len(scene_list)} segments for sharpest frames")
        cap = cv2.VideoCapture(str(video_path))

    candidates: list[dict] = []

    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = int(start_time)
        end_frame = int(end_time)

        # Scan every frame in the scene, track the sharpest
        best_frame_idx = -1
        best_variance = -1.0
        best_img = None

        for f_idx in range(start_frame, min(end_frame, start_frame + 300)):  # cap at 300 frames/10s
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var > best_variance:
                best_variance = laplacian_var
                best_frame_idx = f_idx
                best_img = frame

        if best_frame_idx < 0 or best_variance < blur_variance_min:
            continue  # scene has no sharp frame

        timestamp = best_frame_idx / fps
        out_path = output_dir / f"frame_{best_frame_idx:06d}_{timestamp:.1f}s.jpg"
        cv2.imwrite(str(out_path), best_img)

        candidates.append({
            "frame_idx": best_frame_idx,
            "timestamp_sec": round(timestamp, 1),
            "path": str(out_path.resolve()),
            "sharpness_score": round(best_variance, 1),
        })

    cap.release()

    if verbose:
        print(f"  After sharpness ranking: {len(candidates)} frames")

    if len(candidates) <= 1:
        return candidates[:max_frames]

    # 2. Histogram dedup
    deduped = [candidates[0]]
    prev_hist = None
    for c in candidates[1:]:
        img = cv2.imread(c["path"])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr > dedup_corr_threshold:
                continue  # near-duplicate
        deduped.append(c)
        prev_hist = hist

    if verbose:
        print(f"  After dedup: {len(deduped)} unique frames")

    return deduped[:max_frames]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_keyframes.py <video.mp4> <output_dir>")
        sys.exit(1)
    frames = extract_keyframes(sys.argv[1], sys.argv[2])
    print(f"Extracted {len(frames)} keyframes")
    for f in frames:
        print(f"  [{f['timestamp_sec']}s] {f['path']}  (sharpness={f['sharpness_score']})")
