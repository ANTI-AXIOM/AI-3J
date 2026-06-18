"""
Problem Classifier — damage tracks → insurance problem category + recap
"""

import json
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ProblemClassifier:
    """Maps detected damage tracks → insurance problem category."""

    def __init__(self, config: dict):
        self.config = config
        self.rules = config["problem_rules"]
        self.severity_rules = config["severity"]
        self.classes = config["classes"]
        # build name→id lookup from config
        self.car_class_names = {self.classes[i] for i in config["car_classes"]}
        self.house_class_names = {self.classes[i] for i in config["house_classes"]}

    def classify(self, tracks: list[dict]) -> tuple[str, str, str]:
        """
        Returns: (asset_type, problem_category, severity_level)
        """
        class_names = [t["class_name"] for t in tracks]

        # Detect asset type from classes
        car_count = sum(1 for c in class_names if c in self.car_class_names)
        house_count = sum(1 for c in class_names if c in self.house_class_names)

        if car_count >= house_count and car_count > 0:
            asset = "car"
        elif house_count > 0:
            asset = "house"
        else:
            asset = "unknown"

        # Classify problem
        for problem, rule in self.rules.items():
            matched = sum(1 for c in class_names if c in rule["requires"])
            if matched >= rule["min_classes"]:
                break
        else:
            problem = "unclear"

        # Overall severity
        severity = self._assign_severity(tracks)
        return asset, problem, severity

    def _assign_severity(self, tracks: list[dict]) -> str:
        num_classes = len(set(t["class_name"] for t in tracks))
        high_area = sum(1 for t in tracks if t.get("area_fraction", 0) > 0.20)
        structural_damage = any(
            t["class_name"] in {"structural_crack", "collision_damage", "fire_damage"}
            for t in tracks
        )

        if num_classes >= 3 or structural_damage:
            return "high"
        elif num_classes >= 2 or high_area > 0:
            return "medium"
        return "low"

    def generate_recap(self, tracks: list[dict]) -> dict:
        asset, problem, severity = self.classify(tracks)

        damage_list = []
        for t in tracks:
            damage_list.append({
                "damage": t["class_name"],
                "location": t["location"],
                "confidence": t["avg_confidence"],
                "severity": t["severity"],
            })

        # Build 2-line English summary
        if not damage_list:
            summary = f"Asset identified as {asset}. No damage detected."
        else:
            top = sorted(damage_list, key=lambda d: d["confidence"], reverse=True)[:3]
            items = [f"{d['damage']} on {d['location']} ({d['confidence']:.2f})"
                     for d in top]
            parts = []
            if severity == "high":
                parts.append("Significant damage detected")
            elif severity == "medium":
                parts.append("Moderate damage detected")
            else:
                parts.append("Minor damage detected")

            repair_notes = {
                "collision": "Likely needs bodywork + paint.",
                "water_damage": "Requires drying, mold remediation, and structural check.",
                "fire_damage": "Specialized assessment needed. Possible total loss.",
                "storm_impact": "Structural inspection recommended.",
                "wear_tear": "Cosmetic only — not covered.",
            }
            repair = repair_notes.get(problem, "Assessment needed.")
            summary = f"{asset.capitalize()} — {problem} damage. {'; '.join(items)}. {repair}"

        return {
            "asset": asset,
            "problem": problem,
            "severity": severity,
            "damage_count": len(tracks),
            "tracks": damage_list,
            "summary": summary,
        }


def run_classifier(tracks_json: str, output_json: str, config_path: str = "config.yaml"):
    with open(tracks_json) as f:
        tracks = json.load(f)
    config = load_config(config_path)
    classifier = ProblemClassifier(config)
    recap = classifier.generate_recap(tracks)
    with open(output_json, "w") as f:
        json.dump(recap, f, indent=2)
    print(recap["summary"])
    return recap


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python problem_classifier.py <tracks.json> <output.json> [config.yaml]")
        sys.exit(1)
    config_path = sys.argv[3] if len(sys.argv) > 3 else "config.yaml"
    recap = run_classifier(sys.argv[1], sys.argv[2], config_path)
