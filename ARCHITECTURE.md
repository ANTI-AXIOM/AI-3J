# AI Architecture — Insurance Damage Video Analysis

## 1. Problem Framing

| Question | Answer |
|----------|--------|
| Input | Video file (.mp4/.avi/.mov) submitted with claim |
| Targets | Incident type (car / household), damage objects + location, representative keyframes |
| Tasks | Binary classification (car vs house) + object detection (damage types) + keyframe extraction |
| Success (business) | Reduced processing time, consistent assessment, human-aided triage |
| Success (ML) | Classification F1 ≥ 0.85, detection mAP@0.5 ≥ 0.60, keyframe coverage ≥ 0.90 |
| Constraints | < 200 labeled frames per class, variable lighting/angle/quality, single GPU or CPU |

## 2. Pipeline Overview

```
                        Video Upload
                             │
                             ▼
                    ┌─────────────────┐
                    │  1. KeyFrame    │
                    │    Extraction   │   ← scene-detection + blur rejection + dedup
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  2a. TRAIN   │ │ 2b. INFER    │ │ 2c. EXPORT   │
    │   Branch     │ │   Branch     │ │   Branch     │
    │              │ │              │ │              │
    │ Label →      │ │ YOLOv11 on   │ │ Save best N  │
    │ YOLO finetune│ │ keyframes    │ │ keyframes to │
    │ (offline)    │ │              │ │ doc mgmt sys │
    └──────┬───────┘ └──────┬───────┘ └──────────────┘
           │                │
           │                ▼
           │       ┌─────────────────┐
           │       │  3. Temporal    │
           │       │   Aggregation   │  ← IOU-track across frames
           │       └────────┬────────┘
           │                │
           │                ▼
           │       ┌─────────────────┐
           │       │  4. Problem     │
           │       │   Classifier    │  ← damage classes → 5 insurance categories
           │       └────────┬────────┘
           │                │
           │                ▼
           │       ┌─────────────────┐
           │       │  5. Recap Gen   │  ← JSON + 2-line English summary
           │       └─────────────────┘
           │
           ▼
    ┌──────────────┐
    │ 6. Evaluation│  ← mAP, F1, coverage, failure analysis
    └──────────────┘
```

## 3. Component Details

### 3.1 KeyFrame Extraction (`ingest/extract.py`)
- Scene detection via `scenedetect` (content-aware threshold)
- Blur rejection: Laplacian variance < 100 → discard
- Histogram dedup: correlation > 0.95 → keep only the first
- Output: timestamped `.jpg` files → `/keyframes/`

### 3.2 Training Branch (`train/`)
- **Label format**: COCO JSON → YOLO `.txt` via `ultralytics` converter
- **Model**: `yolo11n.pt` (nano) from Ultralytics — fits CPU inference at 15+ FPS on 640px
- **Augmentation**: mosaic=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10, scale=0.5
- **Split**: video-level (never frame-level) — 70/15/15 train/val/test held-out videos
- **Classes** (car): dent, scratch, crack, broken_glass, collision, wheel_damage, paint_peel
- **Classes** (house): water_damage, fire_damage, structural_crack, roof_damage, broken_window, mold, storm_debris
- **Training**: 100 epochs, batch=8, lr=0.01, patience=20. Expect ~0.60 mAP@0.5 on held-out videos

### 3.3 Inference Branch (`infer/`)
- Run YOLOv11 on every keyframe (or stride=2 for long videos)
- Per frame: `results[0].boxes.xyxy, .cls, .conf` → list of detections
- Stage output: JSON `{frame_idx: [{class, bbox, conf}, ...]}`

### 3.4 Temporal Aggregation (`infer/temporal_agg.py`)
- Simple IOU-track: link detections where IOU(frame[t], frame[t+1]) > 0.3
- Track confidence = mean across frames
- Track class = mode vote across frames
- Remove tracks < 2 frames (spurious)
- Stage output: `[{track_id, class, avg_conf, bbox_sequence, severity}, ...]`

### 3.5 Problem Classifier (`infer/problem_classifier.py`)
Maps damage tracks → insurance problem category via rule matrix:

| Damage Classes Detected | Classified Problem |
|------------------------|--------------------|
| dent, scratch, collision, broken_glass, wheel_damage | Collision (car) |
| water_damage, mold | Water damage |
| fire_damage, soot | Fire damage |
| structural_crack, roof_damage, broken_window, storm_debris | Storm / Impact |
| paint_peel, rust | Wear & Tear (rejected) |
| Any on house + no vehicle damage | Household accident |
| Any on car | Car accident |

Severity heuristics:
- `Low`: one damage class, area < 5% of object, conf > 0.6
- `Medium`: 2 classes or area 5-20%
- `High`: 3+ classes or structural classes present

### 3.6 Recap Generator (`infer/recap.py`)
```json
{
  "asset": "car",
  "problem": "collision",
  "damage_count": 2,
  "tracks": [
    {"damage": "dent", "location": "hood", "confidence": 0.91, "severity": "low"},
    {"damage": "crack", "location": "front_bumper", "confidence": 0.88, "severity": "medium"}
  ],
  "summary": "Car — Collision damage. Front bumper cracked (0.88), hood dent (0.91). Repairable — bodywork + paint."
}
```

### 3.7 Keyframe Export (`export/`)
- Top-5 most representative keyframes (highest avg detection confidence or scene change score)
- Copy to `export/{video_id}/` for document management system integration
- Naming: `{video_id}_frame_{timestamp_sec}s.jpg`

## 4. Dataset Strategy (Limited Data)

| Source | Expected Count | Labeling Effort |
|--------|---------------|-----------------|
| Roboflow Public: car damage | ~150 images | Already labeled |
| Roboflow Public: house damage | ~100 images | Already labeled |
| Web-scraped (Flickr, insurance sites) | ~100 images | ~3 hrs manual labeling |
| Your video keyframes (label subset) | ~50 keyframes | ~2 hrs via CVAT/roboflow |

**Bottom line**: ~400 labeled images across 14 damage classes. Enough for transfer learning + heavy augmentation.

## 5. 3-Day Development Schedule

### Day 1 — Pipeline Skeleton & Data
- [ ] Scaffold repo structure (ingest/, train/, infer/, export/, eval/)
- [ ] KeyFrame extraction script (scenedetect + blur filter + dedup)
- [ ] Download public datasets, label with CVAT → COCO JSON
- [ ] Training branch: data loader, YOLO baseline train

### Day 2 — Inference & Temporal Logic
- [ ] Inference on keyframes → per-frame JSON
- [ ] Temporal aggregation (IOU-track + vote)
- [ ] Problem classifier (rule engine)
- [ ] Recap generator (JSON + English summary)
- [ ] Keyframe export to /export/

### Day 3 — Evaluation & Presentation Prep
- [ ] Evaluate on held-out videos: mAP, F1, coverage, failure cases
- [ ] Document failure analysis in report
- [ ] Build presentation slides:
  - Pipeline diagram (slide 2)
  - Experimental setup (slide 3)
  - Results table + visual examples (slide 4)
  - 3 failure cases + root cause analysis (slide 5)
  - Critical discussion + perspectives (slide 6)

## 6. Evaluation Protocol

| Metric | Target | How |
|--------|--------|-----|
| Classification F1 | ≥ 0.85 | Per-video car vs house |
| Detection mAP@0.5 | ≥ 0.60 | COCO eval on held-out videos |
| Keyframe coverage | ≥ 0.90 | % of damage events with ≥1 keyframe |
| Processing time | ≤ 2 min per 30s video | CPU-only target |
| Recap accuracy | ≥ 0.80 | Human judges: correct problem + damage count |

**No data leakage**: video-level split only. Augmented versions of a training frame stay in training.

## 7. Stack

```
ultralytics     # YOLOv11
scenedetect[opencv]  # keyframe extraction
opencv-python   # image processing
torch           # backbone runtime
numpy, Pillow   # utilities
gradio          # demo UI (bonus)
```

Single `requirements.txt`, runs in WSL or colab. No cloud dependencies.

## 8. Failure Cases to Expect (and Present)

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Miss hairline crack on wall | Resolution too low, crack < 5px | Multi-resolution tiling or SAM refinement |
| Confuse pre-existing scratch with new damage | No temporal context | Flag as "existing" when confidence < threshold |
| Miss damage at night/dark | Low contrast | Adaptive histogram equalization in preprocessing |
| House ≠ car confusion on garage photo | Both visible | CLIP fallback on ambiguous frames |
| Overcount same dent in different frames | Poor IOU track | Require min 3-frame match; degrade overlapping tracks |

## 9. Why Not…

| Approach | Rejected because |
|----------|-----------------|
| Video transformer (TimeSformer, VideoMAE) | Requires ≥10K videos for pretrain; we have ~20 |
| 3D CNN (I3D, C3D) | Overkill for short clips; poor on limited data |
| ViT-based detector (DETR) | Slower, harder to debug, no benefit at our data scale |
| Mask R-CNN (two-stage) | YOLO matches accuracy at 10× speed for our class count |
| End-to-end video-to-recap (LLM) | Opaque, hallucinates, hard to evaluate per component |

YOLOv11n is deliberately chosen for **CPU inference feasibility** (the nano variant runs 15+ FPS on a laptop CPU at 640px) and **debuggability** — each component's output is inspectable.
