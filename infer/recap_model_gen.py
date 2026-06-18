"""
Wraps the trained recap transformer for use in the detection pipeline.
"""

from pathlib import Path
from recap_model import load_recap_model, generate


class RecapModelGenerator:
    """Generates recaps using the trained transformer model."""

    def __init__(self, model_path: str = "models/recap_model.pt", use_beam: bool = True):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.use_beam = use_beam
        self.device = "cpu"

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if not self.available:
            raise FileNotFoundError(
                f"No trained recap model at {self.model_path}. "
                f"Run: python recap_model.py"
            )
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = load_recap_model(str(self.model_path))
        self.model.to(self.device)

    def generate(self, recap_data: dict) -> str:
        if self.model is None:
            self.load()
        return generate(recap_data, self.model, self.tokenizer,
                        use_beam=self.use_beam, device=self.device)
