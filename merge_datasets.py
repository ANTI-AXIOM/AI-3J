"""
Merge car_damage (69 Fr→En), property_damage (28 En), and dataset_frames_old (14)
into one unified dataset — without deleting existing data.
"""

import shutil
from pathlib import Path

CAR_FR_EN = {
    "bosse_aile_arriere_droite": "dent_rear_right_wing",
    "bosse_aile_arriere_gauche": "dent_rear_left_wing",
    "bosse_aile_avant_droite": "dent_front_right_wing",
    "bosse_aile_avant_gauche": "dent_front_left_wing",
    "bosse_capot": "dent_hood",
    "bosse_malle": "dent_trunk",
    "bosse_pare_chocs_arriere": "dent_rear_bumper",
    "bosse_pare_chocs_avant": "dent_front_bumper",
    "bosse_porte_arriere_droite": "dent_rear_right_door",
    "bosse_porte_arriere_gauche": "dent_rear_left_door",
    "bosse_porte_avant_droite": "dent_front_right_door",
    "bosse_porte_avant_gauche": "dent_front_left_door",
    "casse_feu_arriere": "broken_rear_light",
    "casse_pare_brise": "broken_windshield",
    "casse_phare_avant": "broken_headlight",
    "casse_vitre": "broken_window",
    "cassure_capot": "cracked_hood",
    "cassure_pare_chocs_arriere": "cracked_rear_bumper",
    "cassure_pare_chocs_avant": "cracked_front_bumper",
    "cassure_retroviseur": "broken_mirror",
    "ecrasement_aile_arriere_droit": "crushed_rear_right_wing",
    "ecrasement_aile_arriere_gauche": "crushed_rear_left_wing",
    "ecrasement_aile_avant_droite": "crushed_front_right_wing",
    "ecrasement_aile_avant_gauche": "crushed_front_left_wing",
    "ecrasement_capot": "crushed_hood",
    "ecrasement_malle": "crushed_trunk",
    "ecrasement_pare_chocs_arriere": "crushed_rear_bumper",
    "ecrasement_pare_chocs_avant": "crushed_front_bumper",
    "ecrasement_porte_arriere_droite": "crushed_rear_right_door",
    "ecrasement_porte_arriere_gauche": "crushed_rear_left_door",
    "ecrasement_porte_avant": "crushed_front_door",
    "ecrasement_porte_avant_droite": "crushed_front_right_door",
    "ecrasement_porte_avant_gauche": "crushed_front_left_door",
    "ecrasement_toit": "crushed_roof",
    "fissure_aile_arriere_droite": "crack_rear_right_wing",
    "fissure_aile_arriere_gauche": "crack_rear_left_wing",
    "fissure_aile_avant_droite": "crack_front_right_wing",
    "fissure_aile_avant_gauche": "crack_front_left_wing",
    "fissure_capot": "crack_hood",
    "fissure_pare_chocs_arriere": "crack_rear_bumper",
    "fissure_pare_chocs_avant": "crack_front_bumper",
    "fissure_porte_arriere_droite": "crack_rear_right_door",
    "fissure_porte_arriere_gauche": "crack_rear_left_door",
    "fissure_porte_avant_droite": "crack_front_right_door",
    "fissure_porte_avant_gauche": "crack_front_left_door",
    "impact_pare_brise": "impact_windshield",
    "impact_vitre": "impact_window",
    "peinture_ecaillee_aile_arriere_droite": "chipped_paint_rear_right_wing",
    "peinture_ecaillee_aile_arriere_gauche": "chipped_paint_rear_left_wing",
    "peinture_ecaillee_aile_avant_droite": "chipped_paint_front_right_wing",
    "peinture_ecaillee_aile_avant_gauche": "chipped_paint_front_left_wing",
    "peinture_ecaillee_capot": "chipped_paint_hood",
    "peinture_ecaillee_malle": "chipped_paint_trunk",
    "peinture_ecaillee_pare_chocs_arriere": "chipped_paint_rear_bumper",
    "peinture_ecaillee_pare_chocs_avant": "chipped_paint_front_bumper",
    "peinture_ecaillee_porte_arriere_droite": "chipped_paint_rear_right_door",
    "peinture_ecaillee_porte_arriere_gauche": "chipped_paint_rear_left_door",
    "peinture_ecaillee_porte_avant_droite": "chipped_paint_front_right_door",
    "peinture_ecaillee_porte_avant_gauche": "chipped_paint_front_left_door",
    "rayure_aile_arriere_droite": "scratch_rear_right_wing",
    "rayure_aile_arriere_gauche": "scratch_rear_left_wing",
    "rayure_aile_avant_droite": "scratch_front_right_wing",
    "rayure_aile_avant_gauche": "scratch_front_left_wing",
    "rayure_pare_chocs_arriere": "scratch_rear_bumper",
    "rayure_pare_chocs_avant": "scratch_front_bumper",
    "rayure_porte_arriere_droite": "scratch_rear_right_door",
    "rayure_porte_arriere_gauche": "scratch_rear_left_door",
    "rayure_porte_avant_droite": "scratch_front_right_door",
    "rayure_porte_avant_gauche": "scratch_front_left_door",
}

PROPERTY_NAMES = [
    "blown_render_trapped_moisture",
    "brick_algae",
    "brick_efflorescence",
    "broken_or_loose_roof_tile",
    "condensation",
    "damaged_render",
    "flaking_paint_trapped_moisture",
    "lichen_growth",
    "rising_damp",
    "spalled_brickwork",
    "blocked_air_vent",
    "breached_dpc",
    "distortion_to_chimney",
    "drying_out_shrinkage_plaster_interior",
    "drying_out_shrinkage_plaster",
    "internal_cracking",
    "mould_growth",
    "penetrating_or_rising_damp",
    "potential_leak_water_ingress",
    "sill_to_header_cracking_structural",
    "stepped_cracking_structural",
    "structural_cracking",
    "structural_cracking_trapped_moisture",
    "structural_defect_trapped_water",
    "structural_defect_chimney_breast",
    "trapped_moisture",
    "trapped_moisture_chimney_shoulder",
    "trapped_moisture_chimney_stack",
]

UNIFIED_CLASSES = list(CAR_FR_EN.values()) + PROPERTY_NAMES
NC = len(UNIFIED_CLASSES)
print(f"Base unified classes: {NC} (car: 0-{len(CAR_FR_EN)-1}, property: {len(CAR_FR_EN)}-{NC-1})")

# Build car remap
import yaml
with open("car_damage/data.yaml") as f:
    car_cfg = yaml.safe_load(f)
CAR_OLD_NAMES = car_cfg["names"]
CAR_REMAP = {}
for old_id, fr_name in enumerate(CAR_OLD_NAMES):
    en_name = CAR_FR_EN[fr_name]
    new_id = UNIFIED_CLASSES.index(en_name)
    CAR_REMAP[old_id] = new_id

PROP_OFFSET = len(CAR_FR_EN)
PROP_REMAP = {i: PROP_OFFSET + i for i in range(len(PROPERTY_NAMES))}

# Original dataset_frames mapping
ORIG_TO_UNIFIED = {
    0: UNIFIED_CLASSES.index("broken_window"),
    1: UNIFIED_CLASSES.index("broken_window"),
    2: UNIFIED_CLASSES.index("crushed_front_bumper"),
    3: UNIFIED_CLASSES.index("crack_hood"),
    4: UNIFIED_CLASSES.index("dent_hood"),
    7: UNIFIED_CLASSES.index("chipped_paint_hood"),
    9: UNIFIED_CLASSES.index("scratch_front_bumper"),
    13: None,
    5: PROP_OFFSET + PROPERTY_NAMES.index("structural_cracking"),
    6: PROP_OFFSET + PROPERTY_NAMES.index("mould_growth"),
    8: PROP_OFFSET + PROPERTY_NAMES.index("broken_or_loose_roof_tile"),
    10: PROP_OFFSET + PROPERTY_NAMES.index("potential_leak_water_ingress"),
    11: PROP_OFFSET + PROPERTY_NAMES.index("structural_cracking"),
    12: PROP_OFFSET + PROPERTY_NAMES.index("penetrating_or_rising_damp"),
}

# Add extra classes for those not covered
EXTRA_NAMES = []
for cid in [13, 5, 10]:
    if ORIG_TO_UNIFIED[cid] is None or \
       (cid == 5 and ORIG_TO_UNIFIED[cid] == PROP_OFFSET + PROPERTY_NAMES.index("structural_cracking")) or \
       (cid == 10 and ORIG_TO_UNIFIED[cid] == PROP_OFFSET + PROPERTY_NAMES.index("potential_leak_water_ingress")):
        name_map = {13: "wheel_damage", 5: "fire_damage", 10: "storm_debris"}
        EXTRA_NAMES.append(name_map[cid])
        UNIFIED_CLASSES.append(name_map[cid])
        ORIG_TO_UNIFIED[cid] = NC + len(EXTRA_NAMES) - 1
        print(f"  Added extra class: {name_map[cid]} (id={NC + len(EXTRA_NAMES) - 1})")

NC = len(UNIFIED_CLASSES)
print(f"Final unified classes: {NC}")

# ── Copy images + remap labels ──
DST = Path("dataset_frames")
IMGDIR = DST / "images"
LBLDIR = DST / "labels"
IMGDIR.mkdir(parents=True, exist_ok=True)
LBLDIR.mkdir(parents=True, exist_ok=True)

def copy_and_remap(src_img_dir, src_lbl_dir, prefix, remap):
    """Copy images and remapped labels. Skips if image already exists."""
    count_img = 0
    count_lbl = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        new_name = f"{prefix}_{img_path.name}"
        dst_img = IMGDIR / new_name
        if dst_img.exists():
            continue
        shutil.copy2(img_path, dst_img)
        count_img += 1

        stem = img_path.stem
        old_lbl = src_lbl_dir / f"{stem}.txt"
        if old_lbl.exists() and old_lbl.stat().st_size > 0:
            new_lines = []
            with open(old_lbl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    old_id = int(parts[0])
                    if old_id in remap and remap[old_id] is not None:
                        parts[0] = str(remap[old_id])
                        new_lines.append(" ".join(parts))
                        count_lbl += 1
            if new_lines:
                lbl_name = new_name.rsplit(".", 1)[0] + ".txt"
                (LBLDIR / lbl_name).write_text("\n".join(new_lines) + "\n")
    return count_img, count_lbl

# 1. car_damage
for split in ["train", "valid", "test"]:
    imgs, lbls = copy_and_remap(
        Path("car_damage") / split / "images",
        Path("car_damage") / split / "labels",
        "car", CAR_REMAP)
    if imgs:
        print(f"  car_damage/{split}: {imgs} images, {lbls} boxes")

# 2. property_damage
for split in ["train", "valid", "test"]:
    imgs, lbls = copy_and_remap(
        Path("property_damage") / split / "images",
        Path("property_damage") / split / "labels",
        "prop", PROP_REMAP)
    if imgs:
        print(f"  property_damage/{split}: {imgs} images, {lbls} boxes")

# 3. dataset_frames_old (original) — skipped: polluted class space, removed from repo

# Write dataset.yaml
yaml_content = f"""# Unified dataset — car damage + property damage
path: dataset_frames
train: images
val: images

nc: {NC}
names:
"""
for i, name in enumerate(UNIFIED_CLASSES):
    yaml_content += f"  {i}: {name}\n"

Path("dataset.yaml").write_text(yaml_content)

total_imgs = len(list(IMGDIR.iterdir()))
total_lbls = len(list(LBLDIR.iterdir()))
total_boxes = sum(1 for l in LBLDIR.iterdir() if l.stat().st_size > 0)
print(f"\n✓ Done: {total_imgs} images, {total_lbls} label files ({total_boxes} non-empty)")
print(f"  {NC} classes in dataset.yaml")
print(f"  Train: python cli.py train --data dataset.yaml --epochs 100")
