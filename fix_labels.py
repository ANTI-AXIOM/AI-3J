"""
Fix misaligned class IDs in YOLO label files.

In LabelStudio, annotations show the correct class name, but the exported
YOLO label files have wrong numeric IDs due to a mismatched classes.txt
during export.

Usage: python fix_labels.py <remap.json>

remap.json format:
    {"old_id": new_id, ...}
    Example: {"10": 12, "11": 13, "13": 11}
"""

import json
import shutil
import sys
from pathlib import Path


def fix_labels(dataset_root: str, remap: dict[str, int], backup: bool = True):
    dataset_root = Path(dataset_root)
    remap_int = {int(k): v for k, v in remap.items()}

    total_fixed = 0
    total_files = 0

    for split in ["images"]:  # flat structure
        lbl_dir = dataset_root / split.replace("images", "labels")
        if not lbl_dir.exists():
            continue
        for fpath in sorted(lbl_dir.glob("*.txt")):
            if fpath.stat().st_size == 0:
                continue
            original = fpath.read_text().strip()
            lines = original.split("\n")
            new_lines = []
            changed = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id in remap_int:
                    parts[0] = str(remap_int[old_id])
                    changed = True
                new_lines.append(" ".join(parts))
            if changed:
                if backup:
                    shutil.copy2(fpath, fpath.with_suffix(".txt.bak"))
                fpath.write_text("\n".join(new_lines) + "\n")
                total_fixed += 1
            total_files += 1

    print(f"Scanned {total_files} label files, fixed {total_fixed}.")


def print_summary(dataset_root: str):
    """Print current class distribution."""
    from collections import Counter
    import os

    names = {
        0: "broken_glass", 1: "broken_window", 2: "collision_damage",
        3: "crack", 4: "dent", 5: "fire_damage", 6: "mold",
        7: "paint_peel", 8: "roof_damage", 9: "scratch",
        10: "storm_debris", 11: "structural_crack", 12: "water_damage",
        13: "wheel_damage",
    }

    for split in ["images"]:
        lbl_dir = Path(dataset_root) / split.replace("images", "labels")
        counts = Counter()
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            with open(f"{lbl_dir}/{fname}") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        counts[int(line.split()[0])] += 1
        print(f"\nTotal annotations (flat structure):")
        if counts:
            for cid in sorted(counts):
                print(f"  {cid:2d} ({names[cid]:20s}): {counts[cid]:3d}")
        else:
            print("  (no annotations)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary("dataset_frames")
    elif len(sys.argv) > 1:
        remap = json.load(open(sys.argv[1]))
        fix_labels("dataset_frames", remap)
        print("\nAfter fix:")
        print_summary("dataset_frames")
    else:
        print("Usage:")
        print("  python fix_labels.py summary                          # show current distribution")
        print("  python fix_labels.py '{\"10\": 12}'                   # remap class 10 → 12")
        print("  python fix_labels.py fix_remap.json                   # remap from JSON file")
