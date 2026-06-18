"""
Training benchmark logger — dumps per-epoch stats to JSON for analysis.

Usage:
    from benchmark import BenchmarkCallback
    model.add_callback("on_train_epoch_end", BenchmarkCallback(log_path="benchmark.json").on_epoch_end)
"""

import json
import time
import torch
from pathlib import Path


class BenchmarkCallback:
    def __init__(self, log_path: str = "benchmark.json"):
        self.log_path = Path(log_path)
        self.epochs = []
        self._epoch_start = None
        self._batch_times = []
        self._batch_start = None

    def on_batch_start(self, trainer):
        self._batch_start = time.time()

    def on_batch_end(self, trainer):
        if self._batch_start is not None:
            self._batch_times.append(time.time() - self._batch_start)

    def on_epoch_start(self, trainer):
        self._epoch_start = time.time()
        self._batch_times = []

    def on_epoch_end(self, trainer):
        epoch_time = time.time() - self._epoch_start
        epoch = trainer.epoch

        # GPU stats
        gpu_util = 0.0
        gpu_mem = 0
        if torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
            except Exception:
                pass
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB

        avg_batch_time = sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0
        it_per_sec = len(self._batch_times) / epoch_time if epoch_time > 0 else 0

        stats = {
            "epoch": epoch + 1,
            "epoch_time_s": round(epoch_time, 2),
            "avg_batch_time_ms": round(avg_batch_time * 1000, 2),
            "it_per_sec": round(it_per_sec, 2),
            "gpu_util_pct": round(gpu_util, 1),
            "gpu_mem_gb": round(gpu_mem, 2),
            "loss": {},
        }

        # Collect losses from trainer (Tensor or dict)
        if hasattr(trainer, "loss"):
            loss = trainer.loss
            if isinstance(loss, dict):
                for k, v in loss.items():
                    if isinstance(v, (int, float)):
                        stats["loss"][k] = round(float(v), 4)
            elif isinstance(loss, torch.Tensor):
                stats["loss"]["total_loss"] = round(loss.item(), 4)

        # Collect metrics from trainer (dict or logger)
        if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
            for k, v in trainer.metrics.items():
                if isinstance(v, (int, float)):
                    stats[k] = round(float(v), 4)

        # Collect from trainer.validator if available
        if hasattr(trainer, "validator") and trainer.validator is not None:
            val = trainer.validator
            for k in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
                if hasattr(val, k):
                    v = getattr(val, k)
                    if isinstance(v, (int, float)):
                        stats[k] = round(float(v), 4)
            if hasattr(val, "metrics") and isinstance(val.metrics, dict):
                for k, v in val.metrics.items():
                    if isinstance(v, (int, float)) and k not in stats:
                        stats[k] = round(float(v), 4)

        self.epochs.append(stats)

        # Also dump current batch-time stats for the last epoch
        if self._batch_times:
            stats["batch_time_samples"] = [round(t * 1000, 1) for t in self._batch_times[:20]]

        # Write incrementally
        self.log_path.write_text(json.dumps(self.epochs, indent=2))

        # Print summary line
        loss_str = " | ".join(f"{k}={v}" for k, v in stats.get("loss", {}).items())
        print(f"  [BENCH] epoch={stats['epoch']}  {stats['epoch_time_s']}s  "
              f"{stats['it_per_sec']} it/s  GPU={stats['gpu_util_pct']}%  "
              f"mem={stats['gpu_mem_gb']}GB  {loss_str}")

    def get_summary(self) -> dict:
        if not self.epochs:
            return {}
        avg_it = sum(e["it_per_sec"] for e in self.epochs) / len(self.epochs)
        avg_time = sum(e["epoch_time_s"] for e in self.epochs) / len(self.epochs)
        return {
            "total_epochs": len(self.epochs),
            "avg_it_per_sec": round(avg_it, 2),
            "avg_epoch_time_s": round(avg_time, 2),
            "first_epoch_time_s": self.epochs[0]["epoch_time_s"],
            "last_epoch_time_s": self.epochs[-1]["epoch_time_s"],
            "first_epoch_it_per_sec": self.epochs[0]["it_per_sec"],
            "last_epoch_it_per_sec": self.epochs[-1]["it_per_sec"],
            "best_mAP50": max(e.get("mAP50", 0) for e in self.epochs),
        }
