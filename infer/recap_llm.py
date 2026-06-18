"""
Recap generator — uses a local LLM (Ollama) to produce fluent insurance-style summaries.
Falls back to rule-based templates if the LLM is unavailable.
"""

import json
import subprocess
from pathlib import Path


class LLMRecapGenerator:
    """Generates insurance recap text using a local LLM via Ollama."""

    def __init__(self, model: str = "llama3.2:3b", server_url: str = "http://localhost:11434",
                 temperature: float = 0.3, fallback_to_rules: bool = True):
        self.model = model
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.fallback_to_rules = fallback_to_rules
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_ollama()
        return self._available

    def _check_ollama(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.server_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _build_prompt(self, recap_data: dict) -> str:
        """Build a structured prompt from the detection results."""
        asset = recap_data["asset"]
        problem = recap_data["problem"]
        severity = recap_data["severity"]
        tracks = recap_data["tracks"]

        damage_lines = []
        for t in tracks:
            damage_lines.append(
                f"  - {t['damage']} on the {t['location']} "
                f"(confidence: {t['confidence']:.0%}, severity: {t['severity']})"
            )

        prompt = f"""You are an insurance claims assistant. Summarize the following damage assessment for a human supervisor.

Incident type: {asset} accident
Category: {problem}
Overall severity: {severity}
Number of damage areas detected: {len(tracks)}

Damage details:
{chr(10).join(damage_lines)}

Write a concise 2-3 sentence summary in plain English. Include:
1. What happened (type of incident)
2. What specific damage was found (key items)
3. Estimated repair complexity (minor/moderate/significant)
4. Any red flags the adjuster should check

Be factual, professional, and direct. Do not add speculative information."""
        return prompt

    def generate(self, recap_data: dict) -> str:
        """Generate a recap text using the LLM, or fall back to rules."""
        if not self.available:
            if self.fallback_to_rules:
                return self._rule_based_summary(recap_data)
            return "LLM unavailable and fallback disabled."

        prompt = self._build_prompt(recap_data)
        try:
            return self._call_ollama(prompt)
        except Exception as e:
            if self.fallback_to_rules:
                return self._rule_based_summary(recap_data)
            return f"LLM error: {e}"

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and return the generated text."""
        import urllib.request
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": 300,
        }).encode()
        req = urllib.request.Request(
            f"{self.server_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result.get("response", "").strip()

    def _rule_based_summary(self, recap_data: dict) -> str:
        """Fallback rule-based summary (same as original)."""
        asset = recap_data["asset"]
        problem = recap_data["problem"]
        severity = recap_data["severity"]
        tracks = recap_data["tracks"]

        if not tracks:
            return f"Asset identified as {asset}. No damage detected."

        top = sorted(tracks, key=lambda d: d["confidence"], reverse=True)[:3]
        items = [f"{d['damage']} on {d['location']} ({d['confidence']:.0%})"
                 for d in top]

        severity_label = {"high": "Significant", "medium": "Moderate", "low": "Minor"}
        repair_notes = {
            "collision": "Likely needs bodywork + paint.",
            "water_damage": "Requires drying, mold remediation, and structural check.",
            "fire_damage": "Specialized assessment needed. Possible total loss.",
            "storm_impact": "Structural inspection recommended.",
            "wear_tear": "Cosmetic only — not covered.",
        }

        repair = repair_notes.get(problem, "Assessment needed.")
        return f"{severity_label.get(severity, 'Unknown')} {problem} damage on {asset}. {'; '.join(items)}. {repair}"
