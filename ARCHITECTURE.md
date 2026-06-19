# AI Architecture — Insurance Damage Video Analysis

## 1. Problem Framing

| Question | Answer |
|----------|--------|
| Input | Video file (.mp4/.avi/.mov) submitted with claim |
| Targets | Damage objects + location + insurance problem category + human-readable recap |
| Tasks | Keyframe extraction + multi-class object detection (100 damage classes) + temporal aggregation + recap generation |
| Success (ML) | Detection mAP@0.5 ≥ 0.60, recap accuracy ≥ 0.80 |
| Constraints | Single GPU or CPU, no external API dependencies |

## 2. Pipeline Overview

```
                        Video Upload
                             │
                             ▼
                    ┌─────────────────┐
                    │  1. Keyframe    │
                    │   Extraction    │   ← scene-detection + sharpness ranking + dedup
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  2a. TRAIN   │ │ 2b. INFER    │ │ 2c. EXPORT   │
    │   Branch     │ │   Branch     │ │   Branch     │
    │              │ │              │ │              │
    │ YOLO finetune│ │ YOLOv11 on   │ │ Save best N  │
    │ (offline)    │ │ keyframes    │ │ keyframes    │
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
           │       │   Classifier    │  ← rule matrix → 5 categories
           │       └────────┬────────┘
           │                │
           │                ▼
           │       ┌─────────────────┐
           │       │  5. Recap Gen   │  ← trained CausalTransformer ~300K params
           │       └─────────────────┘
           │
           ▼
    ┌──────────────┐
    │ 6. Evaluation│  ← mAP, F1, coverage, failure analysis
    └──────────────┘
```

## 3. Component Details

### 3.1 Keyframe Extraction (`ingest/extract_keyframes.py`)
- Scene detection via `scenedetect` (content-aware threshold)
- Sharpness ranking: per-scene, pick the frame with highest Laplacian variance
- Histogram dedup: correlation > 0.95 → keep only the first
- Output: timestamped `.jpg` files → `--keyframes-dir`

### 3.2 Training Branch (`train/train.py`)
- **Model**: `yolo11n.pt` (nano) transfer-learned to 100 damage classes
- **Dataset**: 2971 images, flat structure (`images/` + `labels/` in `dataset_frames/`), no train/val split
- **Label format**: YOLO `.txt` per image, each line `class_id x_center y_center width height` (normalised 0–1)
- **Classes**: 100 unified classes combining car damage (location-specific: dent_front_bumper, crack_hood, etc.) and property damage (mould, structural_cracking, etc.)
- **Augmentation**: mosaic=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10, scale=0.5
- **Training optimizations**:
  - `freeze=10` — backbone frozen for first 10 epochs, head learns faster
  - `warmup_epochs=5` — gradual LR ramp-up prevents divergence
  - `cos_lr=True` — cosine LR decay avoids plateau
  - `cache=True` — RAM caches images after epoch 1, eliminates disk bottleneck
  - `amp=True` — mixed precision (FP16) cuts GPU compute ~40%
- **Usage**: `python cli.py train --data dataset.yaml --epochs 100 --batch 80 --device 0`

### 3.3 Inference Branch (`infer/detect.py`)
- Run YOLOv11 on every keyframe
- Per frame: `results[0].boxes.xyxy, .cls, .conf` → list of detections
- Stage output: JSON `{frame_idx: [{class, bbox, conf}, ...]}`

### 3.4 Temporal Aggregation (`infer/temporal_agg.py`)
- IOU-track: link detections where IOU(frame[t], frame[t+1]) > 0.3
- Track confidence = mean across frames
- Track class = mode vote across frames
- Remove tracks < 2 frames (spurious)
- Stage output: `[{track_id, class, avg_conf, bbox_sequence, severity}, ...]`

### 3.5 Problem Classifier (`infer/problem_classifier.py`)
Maps damage tracks → insurance problem category via rule matrix:

| Damage Classes Detected | Classified Problem |
|------------------------|--------------------|
| dent, scratch, collision, broken_glass, wheel_damage | Collision |
| water_damage, mould, damp, condensation | Water damage |
| fire_damage, soot | Fire damage |
| structural_cracking, storm_debris, roof_damage | Storm / Impact |
| paint_peel, chipped_paint, wear | Wear & Tear |

### 3.6 Recap Generator (`infer/recap_model_gen.py` + `recap_model.py`)
- **Architecture**: GPT-style causal transformer (self-attention only, no cross-attention)
- **Size**: ~300K parameters (d_model=192, 4 layers, 6 heads, vocab=4000)
- **No external LLM** — fully self-contained, no API calls
- **Training**: 20K procedurally-generated synthetic samples with diverse sentence structures
- **Inference**: ~3ms on CPU, greedy decoding
- **Example output**:
  ```
  "This car has sustained moderate damage across 2 area(s). The most prominent issue is
   a dent on the hood. Additional findings include a cracked front bumper. Structural
   integrity should be verified during repair. Bodywork and paint repair recommended."
  ```

### 3.7 Full Inference Pipeline
Single command:
```bash
python cli.py infer --video raw_dataset/video.mp4 --model models/best.pt \
  --recap-model models/recap_model.pt
```

Output:
```
data/results/
├── 01_keyframes/       # extracted frames
├── 02_detections.json  # per-frame YOLO detections
├── 03_recap.json       # structured recap + summary text
└── 04_export/          # top-5 representative keyframes
```

## 4. Dataset

| Source | Count | Format |
|--------|-------|--------|
| Car damage (French → English remapped) | 1 235 images | YOLO `.txt`, 69 original classes mapped to location-specific English names |
| Property damage | 1 652 images | YOLO `.txt`, 28 English classes |

**Merged**: 2 887 total (2 732 labeled), 100 class IDs (97 active, 3 orphaned from removed original dataset).
**Merge tool**: `merge_datasets.py` handles cross-dataset class ID mapping.

## 5. Stack

```
ultralytics          # YOLOv11
scenedetect[opencv]  # keyframe extraction
opencv-python        # image processing
torch                # YOLO + recap transformer
Pillow, numpy, PyYAML, tqdm
```

Single `requirements.txt`, runs in WSL, native Linux, or colab. No cloud dependencies.

## 6. Training Optimizations

| Technique | Effect |
|-----------|--------|
| `freeze=10` backbone layers | Head learns 3× faster in early epochs |
| `warmup_epochs=5` | Prevents divergence with 100 classes |
| `cos_lr=True` | Smooth LR decay, avoids plateau |
| `cache=True` (RAM caching) | Eliminates disk I/O bottleneck after epoch 1 |
| `amp=True` (mixed precision) | ~40% faster GPU compute |

## 7. Recap Model Architecture

```
Feature vector (84-dim: 33 damages + 44 locations + 7 metadata)
    │
    ▼
Linear(84 → 192)          ← project features to model dimension
    │
    ├── [feat_token] + pos_embed(0)
    ├── [tok_1] + pos_embed(1)
    ├── [tok_2] + pos_embed(2)
    ├── ...
    │
    ▼
TransformerEncoder × 4    ← causal mask, self-attention only
(d_model=192, nhead=6, ff=576)
    │
    ▼
Linear(192 → vocab)       ← predict next token
```

- **Training**: teacher forcing, CrossEntropyLoss, OneCycleLR
- **Inference**: greedy autoregressive, full sequence through encoder each step
- **No external dependencies**: pure PyTorch, no transformers library

## 8. Evaluation Protocol

| Metric | Target | How |
|--------|--------|-----|
| Detection mAP@0.5 | ≥ 0.60 | COCO eval on held-out videos |
| Keyframe coverage | ≥ 0.90 | % of damage events with ≥1 keyframe |
| Processing time | ≤ 2 min per 30s video | CPU target |
| Recap accuracy | ≥ 0.80 | Human judges: correct + fluent |

## 9. Failure Cases

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Miss hairline crack on wall | Resolution too low | Multi-resolution tiling |
| Confuse pre-existing with new damage | No temporal context | Flag low-confidence as "existing" |
| Miss damage at night/dark | Low contrast | Adaptive histogram equalization |
| Overcount same dent in frames | Poor IOU tracking | Min 3-frame match required |
| Recap hallucinates classes not in YOLO output | Model extrapolates | Clamp to detected classes |

## 10. Why Not…

| Approach | Rejected because |
|----------|-----------------|
| Video transformer (TimeSformer, VideoMAE) | Requires ≥10K videos; we have ~20 |
| 3D CNN (I3D, C3D) | Overkill for short clips; poor on limited data |
| ViT-based detector (DETR) | Slower, no benefit at our data scale |
| Mask R-CNN (two-stage) | YOLO matches accuracy at 10× speed |
| End-to-end video-to-recap (LLM) | Opaque, hallucinates, external API dependency |
