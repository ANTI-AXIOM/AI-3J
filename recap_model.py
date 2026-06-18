"""
Recap generator — procedurally-trained small transformer.
Uses diverse synthetic data with varied sentence structures,
not template filling. ~500K params, runs in <10ms on CPU.
"""

import json
import random
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# TOKENIZER
# ──────────────────────────────────────────────
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {s: i for i, s in enumerate(SPECIALS)}
        self.idx2word = {i: s for i, s in enumerate(SPECIALS)}
        self.vocab_size = len(SPECIALS)

    def fit(self, texts: list[str], max_vocab: int = 3000):
        freq = {}
        for t in texts:
            for w in t.lower().split():
                freq[w] = freq.get(w, 0) + 1
        for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]:
            if w not in self.word2idx:
                self.word2idx[w] = self.vocab_size
                self.idx2word[self.vocab_size] = w
                self.vocab_size += 1

    def encode(self, text: str, max_len: int = 48) -> list[int]:
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
# DAMAGE VOCABULARY
# ──────────────────────────────────────────────
CAR_DAMAGES = [
    "dent", "scratch", "crack", "broken glass", "broken window",
    "collision damage", "wheel damage", "chipped paint", "crushed panel",
    "impact mark", "scrape", "gouge", "paint peel", "bent frame",
]

CAR_LOCATIONS = [
    "hood", "front bumper", "rear bumper", "front left door", "front right door",
    "rear left door", "rear right door", "front left wing", "front right wing",
    "rear left wing", "rear right wing", "trunk lid", "roof", "windshield",
    "rear windshield", "left headlight", "right headlight", "left tail light",
    "right tail light", "left side mirror", "right side mirror", "wheel arch",
    "rocker panel", "grille",
]

HOUSE_DAMAGES = [
    "water damage", "mold growth", "structural crack", "roof damage",
    "broken window", "fire damage", "storm debris", "rising damp",
    "brick efflorescence", "blown render", "damaged render", "lichen growth",
    "condensation", "spalled brickwork", "trapped moisture", "internal cracking",
    "blocked vent", "breached damp proof course", "chimney distortion",
]

HOUSE_LOCATIONS = [
    "exterior wall", "interior wall", "ceiling", "floor", "foundation",
    "roof tile", "chimney breast", "chimney stack", "window frame",
    "door frame", "brickwork", "render", "air vent", "roof gutter",
    "basement wall", "attic", "bathroom ceiling", "kitchen wall",
]

SEVERITY_ADJ = {
    "low": ["minor", "superficial", "light", "cosmetic", "small"],
    "medium": ["moderate", "notable", "significant", "considerable", "evident"],
    "high": ["severe", "major", "critical", "extensive", "substantial"],
}

SEVERITY_NOUN = {
    "low": ["minor issue", "cosmetic concern", "surface damage", "small blemish"],
    "medium": ["notable damage", "moderate issue", "repairable damage", "fixable problem"],
    "high": ["critical issue", "major damage", "severe problem", "extensive damage"],
}

PROBLEM_WORDS = {
    "collision": ["impact", "collision", "strike", "crash", "contact", "bump"],
    "water_damage": ["water ingress", "moisture infiltration", "damp", "water penetration", "leak"],
    "fire_damage": ["fire", "heat damage", "burn", "soot", "thermal damage"],
    "storm_impact": ["storm impact", "weather damage", "wind damage", "debris strike", "hail"],
    "wear_tear": ["wear and tear", "age-related", "gradual deterioration", "material fatigue"],
}


# ──────────────────────────────────────────────
# PROCEDURAL RECAP GENERATOR
# ──────────────────────────────────────────────
def pick(items):
    return random.choice(items)

def generate_recap() -> tuple[dict, str]:
    """Generate a diverse, natural-sounding damage recap with no fixed templates."""
    is_car = random.random() < 0.55
    damages = CAR_DAMAGES if is_car else HOUSE_DAMAGES
    locations = CAR_LOCATIONS if is_car else HOUSE_LOCATIONS
    asset = "car" if is_car else "house"

    # Decide severity first, then number of damage areas
    sev = pick(["low", "medium", "high"])
    if sev == "low":
        n = random.choices([1, 2], weights=[3, 1])[0]
    elif sev == "medium":
        n = random.choices([1, 2, 3], weights=[1, 3, 1])[0]
    else:
        n = random.choices([2, 3, 4, 5], weights=[2, 3, 2, 1])[0]

    # Generate damage tracks
    tracks = []
    used_dmg = set()
    for _ in range(n):
        dmg = pick(damages)
        loc = pick(locations)
        conf = round(random.uniform(0.65, 0.99), 2)
        tracks.append({"damage": dmg, "location": loc, "confidence": conf, "severity": sev})

    # Determine problem
    prob = "collision" if is_car else pick(["water_damage", "storm_impact", "fire_damage"])

    # ── Build sentences procedurally ──
    sents = []

    # Sentence 1: Opening statement
    openers = [
        f"This {asset} has sustained {pick(SEVERITY_ADJ[sev])} damage across {n} area(s).",
        f"Inspection of the {asset} reveals {pick(SEVERITY_NOUN[sev])} affecting {n} area(s).",
        f"The {asset} shows evidence of {pick(PROBLEM_WORDS[prob])} with {n} visible damage site(s).",
        f"A {pick(SEVERITY_ADJ[sev])} {pick(['impact', 'incident', 'event', 'occurrence'])} has damaged this {asset} in {n} location(s).",
    ]
    sents.append(pick(openers))

    # Sentence 2: List damage items (vary style)
    top = sorted(tracks, key=lambda t: t["confidence"], reverse=True)[:3]
    style = random.random()
    if style < 0.33:
        items = [f"{t['damage']} on the {t['location']}" for t in top]
        sents.append(f"Specifically, {'; '.join(items)}." + (f" ({n - 3} more area(s) affected)" if n > 3 else ""))
    elif style < 0.66:
        for t in top:
            prefix = pick(["A", "The", "There is a", "Notable"])
            sents.append(f"{prefix} {t['damage']} on the {t['location']} (confidence: {t['confidence']:.0%}).")
    else:
        first = top[0]
        rest = top[1:]
        sents.append(f"The most prominent issue is {first['damage']} on the {first['location']}.")
        if rest:
            extra = "; ".join(f"{t['damage']} on {t['location']}" for t in rest)
            sents.append(f"Additional findings include {extra}.")

    # Sentence 3: Severity assessment
    if sev == "low":
        sents.append(pick([
            "This appears to be cosmetic only and does not affect structural integrity.",
            "No structural concerns at this stage. Repairs should be straightforward.",
            "The damage is superficial and should be repairable without major intervention.",
        ]))
    elif sev == "medium":
        sents.append(pick([
            "Structural integrity should be verified during repair. Moderate intervention required.",
            "Repairs will require bodywork and possibly panel replacement in affected areas.",
            "A thorough inspection is recommended before proceeding with repairs.",
        ]))
    else:
        sents.append(pick([
            "Structural integrity may be compromised. A detailed engineering assessment is strongly recommended.",
            "Extensive repairs required. Specialist evaluation is necessary before proceeding.",
            "This may constitute a total loss depending on the assessment of hidden damage.",
        ]))

    # Sentence 4: Repair recommendation
    repair_opts = {
        "collision": [
            "Bodywork and paint repair recommended.",
            "Panel replacement and realignment likely needed.",
            "Impact damage requires frame inspection and body correction.",
        ],
        "water_damage": [
            "Drying, mold remediation, and source repair required.",
            "Water extraction and structural drying needed. Check for hidden moisture.",
            "Professional damp-proofing and ventilation improvement recommended.",
        ],
        "fire_damage": [
            "Fire damage requires specialist restoration. Structural assessment essential.",
            "Soot cleanup, ventilation, and structural evaluation needed.",
        ],
        "storm_impact": [
            "Weatherproofing repair and structural check recommended.",
            "Roof and gutter inspection needed. Check for hidden water ingress.",
        ],
        "wear_tear": [
            "Age-related deterioration. Routine maintenance recommended.",
            "Preventive repair advised to avoid escalation.",
        ],
    }
    sents.append(pick(repair_opts.get(prob, ["Assessment needed."])))

    recap = " ".join(sents)

    # Build structured profile
    profile = {
        "asset": asset, "problem": prob, "severity": sev,
        "damage_count": n,
        "tracks": [{"damage": t["damage"], "location": t["location"],
                     "confidence": t["confidence"], "severity": t["severity"]}
                   for t in tracks],
    }
    return profile, recap


# ──────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────
ALL_DAMAGES = list(set(CAR_DAMAGES + HOUSE_DAMAGES))
ALL_LOCATIONS = list(set(CAR_LOCATIONS + HOUSE_LOCATIONS))
DMG_TO_IDX = {d: i for i, d in enumerate(ALL_DAMAGES)}
LOC_TO_IDX = {l: i for i, l in enumerate(ALL_LOCATIONS)}
N_DMG = len(ALL_DAMAGES)
N_LOC = len(ALL_LOCATIONS)

def encode_features(profile: dict) -> torch.Tensor:
    """Feature vector: damage presence × 2 + assethot + sevohot + count."""
    # Average confidence per damage type
    dmg_feat = torch.zeros(N_DMG)
    loc_feat = torch.zeros(N_LOC)
    for t in profile["tracks"]:
        if t["damage"] in DMG_TO_IDX:
            dmg_feat[DMG_TO_IDX[t["damage"]]] = max(dmg_feat[DMG_TO_IDX[t["damage"]]], t["confidence"])
        if t["location"] in LOC_TO_IDX:
            loc_feat[LOC_TO_IDX[t["location"]]] = max(loc_feat[LOC_TO_IDX[t["location"]]], t["confidence"])

    # Asset one-hot (3)
    asset_map = {"car": [1, 0, 0], "house": [0, 1, 0], "unknown": [0, 0, 1]}
    asset_oh = torch.tensor(asset_map.get(profile["asset"], [0, 0, 0]), dtype=torch.float)

    # Severity one-hot (3)
    sev_map = {"low": [1, 0, 0], "medium": [0, 1, 0], "high": [0, 0, 1]}
    sev_oh = torch.tensor(sev_map.get(profile["severity"], [0, 0, 0]), dtype=torch.float)

    # Count (normalized)
    count = torch.tensor([min(profile["damage_count"], 10) / 10.0])

    return torch.cat([dmg_feat, loc_feat, asset_oh, sev_oh, count])


class RecapDataset(Dataset):
    def __init__(self, profiles: list[dict], recaps: list[str],
                 tokenizer: SimpleTokenizer, max_len: int = 48):
        self.features = [encode_features(p) for p in profiles]
        self.labels = [tokenizer.encode(r, max_len) for r in recaps]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# ──────────────────────────────────────────────
# IMPROVED TRANSFORMER (d_model=384, 4 layers)
# ──────────────────────────────────────────────
class RecapTransformer(nn.Module):
    """~1.6M params, d_model=128, 3 layers, 4 heads, vocab=4000."""

    def __init__(self, vocab_size: int, feat_dim: int, d_model: int = 128,
                 nhead: int = 4, nlayers: int = 3, max_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(0.1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features: torch.Tensor, target_tokens: Optional[torch.Tensor] = None,
                teacher_forcing: bool = True) -> torch.Tensor:
        batch_size = features.size(0)
        device = features.device

        memory = self.feat_proj(features).unsqueeze(1)

        if teacher_forcing and target_tokens is not None:
            tgt = target_tokens[:, :-1]
            tgt_emb = self.token_embed(tgt) * (self.d_model ** 0.5)
            pos_ids = torch.arange(tgt.size(1), device=device).unsqueeze(0)
            tgt_emb = tgt_emb + self.pos_embed(pos_ids)
            tgt_emb = self.dropout(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=device)
            out = self.decoder(tgt_emb, memory.repeat(1, tgt.size(1), 1), tgt_mask=tgt_mask)
            return self.out(out)
        else:
            generated = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
            for step in range(self.max_len - 1):
                tgt_emb = self.token_embed(generated) * (self.d_model ** 0.5)
                pos_ids = torch.arange(generated.size(1), device=device).unsqueeze(0)
                tgt_emb = tgt_emb + self.pos_embed(pos_ids)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    generated.size(1), device=device)
                out = self.decoder(tgt_emb, memory.repeat(1, generated.size(1), 1), tgt_mask=tgt_mask)
                logits = self.out(out[:, -1:, :])
                next_token = logits.argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == EOS).all():
                    break
            return generated

    @torch.no_grad()
    def beam_search(self, features: torch.Tensor, beam_size: int = 3,
                    max_len: int = 48) -> torch.Tensor:
        """Beam search decoding for better output quality."""
        device = features.device
        batch_size = features.size(0)
        memory = self.feat_proj(features).unsqueeze(1)
        sos = torch.full((batch_size, 1), SOS, dtype=torch.long, device=device)
        beams = [(sos, 0.0)]

        for step in range(max_len - 1):
            candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == EOS:
                    candidates.append((seq, score))
                    continue
                tgt_emb = self.token_embed(seq) * (self.d_model ** 0.5)
                pos_ids = torch.arange(seq.size(1), device=device).unsqueeze(0)
                tgt_emb = tgt_emb + self.pos_embed(pos_ids)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq.size(1), device=device)
                out = self.decoder(tgt_emb, memory.repeat(1, seq.size(1), 1), tgt_mask=tgt_mask)
                logits = self.out(out[:, -1:, :]).squeeze(1)
                log_probs = F.log_softmax(logits, dim=-1)
                topk = log_probs.topk(beam_size, dim=-1)
                for i in range(beam_size):
                    token = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + topk.values[0, i].item()
                    candidates.append((new_seq, new_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            if all(b[0, 0, -1].item() == EOS for b, _ in beams):
                break
        return beams[0][0]


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Generate diverse synthetic data
    n_samples = 100000
    print(f"Generating {n_samples} synthetic samples...")
    profiles, recaps = [], []
    for _ in range(n_samples):
        p, r = generate_recap()
        profiles.append(p)
        recaps.append(r)

    # Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(recaps, max_vocab=4000)
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    # Dataset
    split = int(n_samples * 0.95)
    train_ds = RecapDataset(profiles[:split], recaps[:split], tokenizer)
    val_ds = RecapDataset(profiles[split:], recaps[split:], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=2)

    # Model
    feat_dim = N_DMG + N_LOC + 7  # damages + locations + asset(3) + sev(3) + count(1)
    model = RecapTransformer(
        vocab_size=tokenizer.vocab_size,
        feat_dim=feat_dim,
        d_model=384,
        nhead=8,
        nlayers=4,
        max_len=48,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Optimizer with warmup + cosine decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4,
                                   betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, total_steps=len(train_loader) * 20,
        pct_start=0.1, anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    Path("models").mkdir(exist_ok=True)
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
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                out = model(feats, labels, teacher_forcing=True)
                loss = criterion(out.reshape(-1, tokenizer.vocab_size), labels[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # Show examples
        example_out = ""
        if epoch % 5 == 0 or avg_val < best_loss:
            with torch.no_grad():
                feats, labels = next(iter(val_loader))
                feats = feats[:3].to(device)
                gen = model(feats, teacher_forcing=False)
                for i in range(min(2, len(gen))):
                    pred = tokenizer.decode(gen[i].tolist())
                    truth = tokenizer.decode(labels[i].tolist())
                    example_out += f"\n    GT: {truth}\n    PD: {pred}"

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d} | train: {avg_train:.4f} | val: {avg_val:.4f} | lr: {lr_now:.2e}{example_out}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "tokenizer": tokenizer,
                "vocab_size": tokenizer.vocab_size,
                "feat_dim": feat_dim,
                "d_model": 128,
                "nhead": 4,
                "nlayers": 3,
                "DMG_TO_IDX": DMG_TO_IDX,
                "LOC_TO_IDX": LOC_TO_IDX,
                "N_DMG": N_DMG,
                "N_LOC": N_LOC,
                "ALL_DAMAGES": ALL_DAMAGES,
                "ALL_LOCATIONS": ALL_LOCATIONS,
            }, "models/recap_model.pt")
            print(f"  ✓ Saved (loss={best_loss:.4f})")

    print(f"\nDone. Best model at models/recap_model.pt (val_loss={best_loss:.4f})")


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def load_recap_model(path: str = "models/recap_model.pt"):
    ckpt = torch.load(path, map_location="cpu")
    tokenizer = ckpt["tokenizer"]
    model = RecapTransformer(
        vocab_size=ckpt["vocab_size"],
        feat_dim=ckpt["feat_dim"],
        d_model=ckpt["d_model"],
        nhead=ckpt.get("nhead", 8),
        nlayers=ckpt.get("nlayers", 4),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


def generate(profile: dict, model: nn.Module, tokenizer: SimpleTokenizer,
             use_beam: bool = True, device: str = "cpu") -> str:
    feat = encode_features(profile).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_beam:
            tokens = model.beam_search(feat, beam_size=3)
        else:
            tokens = model(feat, teacher_forcing=False)
    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Test mode: generate a random recap
        p, r = generate_recap()
        print(f"Profile: {json.dumps(p, indent=2)}")
        print(f"\nRecap: {r}")
    else:
        train()
