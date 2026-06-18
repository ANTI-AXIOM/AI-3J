"""
Trainable recap generator — small transformer that learns to produce
insurance summaries from structured damage features.

Architecture: 2-layer transformer encoder + decoder, ~50K parameters.
Trained on synthetic data generated from templates with randomization,
so the model generalizes beyond any single template pattern.

Usage:
    python train_recap.py                          # train the model
    python infer_recap.py --track-json tracks.json  # generate recap
"""

import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# 1. TOKENIZER (simple word-level)
# ──────────────────────────────────────────────
PAD = 0
SOS = 1
EOS = 2
UNK = 3
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

class SimpleTokenizer:
    """Word-level tokenizer built from training data."""

    def __init__(self):
        self.word2idx = {s: i for i, s in enumerate(SPECIALS)}
        self.idx2word = {i: s for i, s in enumerate(SPECIALS)}
        self.vocab_size = len(SPECIALS)

    def fit(self, texts: list[str], max_vocab: int = 2000):
        freq = {}
        for t in texts:
            for w in t.lower().split():
                freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq, key=freq.get, reverse=True)[:max_vocab]
        for w in sorted_words:
            if w not in self.word2idx:
                idx = self.vocab_size
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                self.vocab_size += 1

    def encode(self, text: str, max_len: int = 40) -> list[int]:
        tokens = [SOS]
        for w in text.lower().split()[:max_len - 2]:
            tokens.append(self.word2idx.get(w, UNK))
        tokens.append(EOS)
        tokens += [PAD] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, tokens: list[int]) -> str:
        words = []
        for t in tokens:
            if t == EOS:
                break
            if t > 3:
                words.append(self.idx2word.get(t, "<UNK>"))
        return " ".join(words)


# ──────────────────────────────────────────────
# 2. SYNTHETIC TRAINING DATA
# ──────────────────────────────────────────────
CAR_CLASSES = [
    "dent_hood", "dent_front_bumper", "dent_rear_bumper", "dent_front_left_door",
    "dent_front_right_door", "dent_rear_left_door", "dent_rear_right_door",
    "dent_front_left_wing", "dent_front_right_wing", "dent_rear_left_wing",
    "dent_rear_right_wing", "dent_trunk",
    "scratch_front_bumper", "scratch_rear_bumper", "scratch_front_left_door",
    "scratch_front_right_door", "scratch_rear_left_door", "scratch_rear_right_door",
    "scratch_front_left_wing", "scratch_front_right_wing", "scratch_rear_left_wing",
    "scratch_rear_right_wing",
    "crack_hood", "crack_front_bumper", "crack_rear_bumper", "crack_rear_right_wing",
    "crack_rear_left_wing", "crack_front_right_wing", "crack_front_left_wing",
    "crack_front_right_door", "crack_front_left_door", "crack_rear_right_door",
    "crack_rear_left_door",
    "broken_window", "broken_windshield", "broken_headlight", "broken_rear_light",
    "broken_mirror",
    "crushed_front_bumper", "crushed_rear_bumper", "crushed_hood", "crushed_trunk",
    "crushed_front_left_wing", "crushed_front_right_wing", "crushed_roof",
    "chipped_paint_hood", "chipped_paint_front_bumper", "chipped_paint_front_left_door",
    "wheel_damage",
]

HOUSE_CLASSES = [
    "blown_render_trapped_moisture", "brick_algae", "brick_efflorescence",
    "broken_or_loose_roof_tile", "condensation", "damaged_render",
    "flaking_paint_trapped_moisture", "lichen_growth", "rising_damp",
    "spalled_brickwork", "blocked_air_vent", "breached_dpc",
    "distortion_to_chimney", "mould_growth", "structural_cracking",
    "internal_cracking", "penetrating_or_rising_damp", "potential_leak_water_ingress",
    "fire_damage", "storm_debris",
]

LOCATIONS = [
    "hood", "front bumper", "rear bumper", "front left door", "front right door",
    "rear left door", "rear right door", "front left wing", "front right wing",
    "rear left wing", "rear right wing", "trunk", "roof", "windshield",
    "ceiling", "wall", "floor", "foundation", "chimney", "window",
    "exterior wall", "roof tile", "brickwork",
]

SEVERITY = ["low", "medium", "high"]

TEMPLATES = [
    # asset + damage summary templates
    "{asset} damage identified: {count} area(s) affected. {dmg_list}. Severity: {severity}.",
    "Inspection found {count} damage area(s) on the {asset}. {dmg_list}. Overall severity is {severity}.",
    "{asset} shows {count} visible damage(s). {dmg_list}. Assessment: {severity} severity.",
    # problem-specific
    "Collision damage to the {asset}. {dmg_list}. Bodywork and paint assessment needed.",
    "Water damage detected on the {asset}. {dmg_list}. Drying and mold inspection recommended.",
    "Fire damage on the {asset}. {dmg_list}. Specialist assessment required.",
    "Structural damage found on the {asset}. {dmg_list}. Engineering inspection advised.",
    "Storm impact damage on the {asset}. {dmg_list}. Check for hidden structural issues.",
    # severity-specific
    "Minor cosmetic damage on the {asset}. {dmg_list}. No structural concerns.",
    "Moderate damage to the {asset}. {dmg_list}. Repair recommended.",
    "Significant damage to the {asset}. {dmg_list}. Detailed assessment required.",
]

REPAIR_NOTES = [
    "Bodywork and paint required.",
    "Panel replacement likely needed.",
    "Drying and mold remediation needed.",
    "Structural repair recommended.",
    "Cosmetic only — no immediate repair needed.",
    "Specialized assessment recommended.",
    "Possible total loss — further inspection needed.",
    "Check for hidden moisture damage.",
    "Minor repair sufficient.",
    "Full replacement recommended.",
]

def generate_synthetic_sample() -> tuple[dict, str]:
    """Generate a structured damage profile + its recap text."""
    is_car = random.random() < 0.6
    classes = CAR_CLASSES if is_car else HOUSE_CLASSES
    asset = "car" if is_car else "house"

    n_damages = random.randint(0, 5)
    tracks = []
    for i in range(n_damages):
        tracks.append({
            "class_name": random.choice(classes),
            "location": random.choice(LOCATIONS),
            "confidence": round(random.uniform(0.6, 0.99), 2),
            "severity": random.choice(SEVERITY),
        })

    if not tracks:
        recap = f"Asset identified as {asset}. No damage detected."
        return {"asset": asset, "problem": "none", "severity": "low",
                "damage_count": 0, "tracks": []}, recap

    # Determine problem type
    car_problems = {"dent": "collision", "scratch": "collision", "crack": "collision",
                    "crushed": "collision", "broken": "collision", "chipped": "collision",
                    "wheel": "collision"}
    house_problems = {"water": "water_damage", "moisture": "water_damage", "damp": "water_damage",
                      "mould": "water_damage", "fire": "fire_damage", "storm": "storm_impact",
                      "structural": "storm_impact", "cracking": "storm_impact"}
    damage_classes = [t["class_name"] for t in tracks]
    top_dmg = damage_classes[0]
    if is_car:
        problem = "collision"
    else:
        for key, prob in house_problems.items():
            if key in top_dmg:
                problem = prob
                break
        else:
            problem = "water_damage"

    severity = random.choice(SEVERITY)

    # Build damage list text
    dmg_items = []
    for t in tracks[:3]:
        dmg_items.append(f"{t['class_name'].replace('_', ' ')} on {t['location']}")
    dmg_list = "; ".join(dmg_items)
    if len(tracks) > 3:
        dmg_list += f", and {len(tracks) - 3} other area(s)"

    # Fill template
    template = random.choice(TEMPLATES)
    try:
        recap = template.format(asset=asset, count=n_damages,
                                 dmg_list=dmg_list, severity=severity)
    except KeyError:
        recap = f"{severity.capitalize()} {problem} damage on {asset}. {dmg_list}."

    # Add repair note
    recap += " " + random.choice(REPAIR_NOTES)

    damage_list = [{"damage": t["class_name"], "location": t["location"],
                    "confidence": t["confidence"], "severity": t["severity"]}
                   for t in tracks]
    return {
        "asset": asset,
        "problem": problem,
        "severity": severity,
        "damage_count": n_damages,
        "tracks": damage_list,
    }, recap


# ──────────────────────────────────────────────
# 3. FEATURE ENCODER (damage → vector)
# ──────────────────────────────────────────────
ALL_CLASSES = list(set(CAR_CLASSES + HOUSE_CLASSES))
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
N_CLASSES = len(ALL_CLASSES)
N_LOCATIONS = len(LOCATIONS)
LOC_TO_IDX = {l: i for i, l in enumerate(LOCATIONS)}

def encode_features(profile: dict) -> torch.Tensor:
    """Convert damage profile to feature vector [N_CLASSES + 3]."""
    feat = torch.zeros(N_CLASSES + 3)
    # Asset: car=1, house=2, unknown=0
    asset_map = {"car": 1, "house": 2, "unknown": 0}
    feat[N_CLASSES] = asset_map.get(profile["asset"], 0)
    # Severity: low=0, medium=1, high=2
    sev_map = {"low": 0, "medium": 1, "high": 2}
    feat[N_CLASSES + 1] = sev_map.get(profile["severity"], 0) / 2.0
    # Damage count (normalized)
    feat[N_CLASSES + 2] = min(profile["damage_count"], 10) / 10.0
    # Damage types: average confidence per class
    for t in profile["tracks"]:
        cname = t["damage"]
        if cname in CLASS_TO_IDX:
            feat[CLASS_TO_IDX[cname]] = t["confidence"]
    return feat


class RecapDataset(Dataset):
    def __init__(self, profiles: list[dict], recaps: list[str],
                 tokenizer: SimpleTokenizer, max_len: int = 40):
        self.features = [encode_features(p) for p in profiles]
        self.labels = [tokenizer.encode(r, max_len) for r in recaps]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# ──────────────────────────────────────────────
# 4. SMALL TRANSFORMER MODEL
# ──────────────────────────────────────────────
class RecapTransformer(nn.Module):
    """Small transformer: ~50K params, 2 layers, 4 heads, d_model=128."""

    def __init__(self, vocab_size: int, feat_dim: int, d_model: int = 128,
                 nhead: int = 4, nlayers: int = 2, max_len: int = 40):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Feature projection
        self.feat_proj = nn.Linear(feat_dim, d_model)

        # Token embedding + position
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer decoder (takes concatenated features + tokens)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=256, dropout=0.1,
                                       batch_first=True),
            num_layers=nlayers
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, target_tokens: torch.Tensor | None = None,
                teacher_forcing: bool = True) -> torch.Tensor:
        batch_size = features.size(0)
        device = features.device

        # Encode features → memory (1 token's worth)
        memory = self.feat_proj(features).unsqueeze(1)  # (B, 1, d_model)

        if teacher_forcing and target_tokens is not None:
            # Use ground truth tokens
            tgt = target_tokens[:, :-1]
            tgt_emb = self.token_embed(tgt)
            pos_ids = torch.arange(tgt.size(1), device=device).unsqueeze(0)
            tgt_emb = tgt_emb + self.pos_embed(pos_ids)

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=device)

            out = self.decoder(tgt_emb, memory.repeat(1, tgt.size(1), 1),
                               tgt_mask=tgt_mask)
            return self.out(out)  # (B, seq_len, vocab)
        else:
            # Autoregressive generation
            sos = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
            generated = sos
            for step in range(self.max_len - 1):
                tgt_emb = self.token_embed(generated)
                pos_ids = torch.arange(generated.size(1), device=device).unsqueeze(0)
                tgt_emb = tgt_emb + self.pos_embed(pos_ids)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    generated.size(1), device=device)
                mem = memory.repeat(1, generated.size(1), 1)

                out = self.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
                logits = self.out(out[:, -1:, :])  # last token only
                next_token = logits.argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all EOS
                if (next_token == EOS).all():
                    break
            return generated


# ──────────────────────────────────────────────
# 5. TRAINING
# ──────────────────────────────────────────────
def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Generate synthetic data
    n_samples = 50000
    print(f"Generating {n_samples} synthetic samples...")
    profiles = []
    recaps = []
    for _ in range(n_samples):
        p, r = generate_synthetic_sample()
        profiles.append(p)
        recaps.append(r)

    # Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(recaps)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Dataset
    split = int(n_samples * 0.9)
    train_ds = RecapDataset(profiles[:split], recaps[:split], tokenizer)
    val_ds = RecapDataset(profiles[split:], recaps[split:], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    feat_dim = N_CLASSES + 3
    model = RecapTransformer(
        vocab_size=tokenizer.vocab_size,
        feat_dim=feat_dim,
        d_model=128,
        max_len=40
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_loss = float("inf")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(feats, labels, teacher_forcing=True)
            loss = criterion(out.reshape(-1, tokenizer.vocab_size), labels[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                out = model(feats, labels, teacher_forcing=True)
                loss = criterion(out.reshape(-1, tokenizer.vocab_size), labels[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d} | train loss: {avg_train:.4f} | val loss: {avg_val:.4f}")

        # Save best model
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "tokenizer": tokenizer,
                "vocab_size": tokenizer.vocab_size,
                "feat_dim": feat_dim,
                "d_model": 128,
                "CLASS_TO_IDX": CLASS_TO_IDX,
                "LOC_TO_IDX": LOC_TO_IDX,
                "N_CLASSES": N_CLASSES,
                "ALL_CLASSES": ALL_CLASSES,
                "LOCATIONS": LOCATIONS,
            }, "models/recap_model.pt")
            print(f"  → Saved best model (loss={best_loss:.4f})")

    # Show examples
    print("\nExamples:")
    model.eval()
    with torch.no_grad():
        for i in range(3):
            idx = random.randint(0, len(val_ds) - 1)
            feat, label = val_ds[idx]
            feat = feat.unsqueeze(0).to(device)
            tokens = model(feat, teacher_forcing=False)
            pred = tokenizer.decode(tokens[0].tolist())
            truth = tokenizer.decode(label.tolist())
            print(f"  GT:  {truth}")
            print(f"  PRED: {pred}")
            print()

    print(f"Done. Model saved to models/recap_model.pt")


# ──────────────────────────────────────────────
# 6. INFERENCE
# ──────────────────────────────────────────────
def load_recap_model(path: str = "models/recap_model.pt") -> tuple:
    """Load trained model + tokenizer."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Run train_recap.py first.")
    checkpoint = torch.load(path, map_location="cpu")
    tokenizer = checkpoint["tokenizer"]
    model = RecapTransformer(
        vocab_size=checkpoint["vocab_size"],
        feat_dim=checkpoint["feat_dim"],
        d_model=checkpoint["d_model"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint


def generate_recap(profile: dict, model: nn.Module, tokenizer: SimpleTokenizer,
                   device: str = "cpu") -> str:
    """Generate a recap for a damage profile."""
    feat = encode_features(profile).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model(feat, teacher_forcing=False)
    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    train()
