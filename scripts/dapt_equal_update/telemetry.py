"""Exact, resume-safe training telemetry for Experiment D MLM runs.

The tracker stores non-padding tokens by completed optimizer step.  Entries are
keyed by global step, so resuming from an earlier checkpoint overwrites rerun
steps instead of double-counting them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import Trainer, TrainerCallback


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    os.replace(temporary, path)


class TrainingTelemetry(TrainerCallback):
    """Collect loss/LR logs and exact non-padding tokens per completed step."""

    def __init__(self, path: str | Path, configured: dict):
        self.path = Path(path)
        self.configured = configured
        self.pending_tokens = 0
        self.tokens_by_step: dict[str, int] = {}
        self.logs_by_step: dict[str, dict] = {}
        self.final_learning_rate_after_training = None
        if self.path.exists():
            prior = json.loads(self.path.read_text(encoding="utf8"))
            self.tokens_by_step = {
                str(key): int(value)
                for key, value in prior.get("tokens_by_step", {}).items()
            }
            self.logs_by_step = {
                str(key): value for key, value in prior.get("logs_by_step", {}).items()
            }
            self.final_learning_rate_after_training = prior.get(
                "final_learning_rate_after_training"
            )

    def observe_batch(self, inputs: dict) -> None:
        mask = inputs.get("attention_mask")
        if mask is None:
            raise RuntimeError("attention_mask missing; cannot count processed tokens")
        self.pending_tokens += int(mask.detach().sum().item())

    def _write(self, completed_steps: int) -> None:
        ordered_logs = [
            {"step": int(step), **values}
            for step, values in sorted(
                self.logs_by_step.items(), key=lambda pair: int(pair[0])
            )
        ]
        optimizer_learning_rates = [
            float(row["optimizer_learning_rate"])
            for row in ordered_logs if "optimizer_learning_rate" in row
        ]
        payload = {
            "schema_version": 1,
            "configured": self.configured,
            "completed_optimizer_updates": completed_steps,
            "total_non_padding_tokens_processed": sum(self.tokens_by_step.values()),
            "tokens_by_step": self.tokens_by_step,
            "logs_by_step": self.logs_by_step,
            "loss_lr_trajectory": ordered_logs,
            "observed_peak_learning_rate": (
                max(optimizer_learning_rates) if optimizer_learning_rates else None
            ),
            "observed_final_learning_rate": self.final_learning_rate_after_training,
            "final_learning_rate_after_training": self.final_learning_rate_after_training,
        }
        _atomic_json(self.path, payload)

    def on_train_begin(self, args, state, control, **kwargs):
        # A valid Trainer checkpoint is authoritative after interruption.
        current = int(state.global_step)
        self.tokens_by_step = {
            step: value for step, value in self.tokens_by_step.items()
            if int(step) <= current
        }
        self.logs_by_step = {
            step: value for step, value in self.logs_by_step.items()
            if int(step) <= current
        }
        self.pending_tokens = 0
        self._write(current)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = str(int(state.global_step))
        self.tokens_by_step[step] = int(self.pending_tokens)
        self.pending_tokens = 0
        optimizer = kwargs.get("optimizer")
        if optimizer is not None and optimizer.param_groups:
            self.logs_by_step.setdefault(step, {})["optimizer_learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )
        self._write(int(state.global_step))
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            keep = {
                key: float(value)
                for key, value in logs.items()
                if key in {"loss", "learning_rate", "grad_norm", "epoch"}
                and isinstance(value, (int, float))
            }
            if keep:
                self.logs_by_step.setdefault(str(int(state.global_step)), {}).update(keep)
                self._write(int(state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        scheduler = kwargs.get("lr_scheduler")
        optimizer = kwargs.get("optimizer")
        if scheduler is not None:
            rates = scheduler.get_last_lr()
            self.final_learning_rate_after_training = float(rates[0]) if rates else None
        elif optimizer is not None and optimizer.param_groups:
            self.final_learning_rate_after_training = float(optimizer.param_groups[0]["lr"])
        self._write(int(state.global_step))
        return control


class TokenCountingTrainer(Trainer):
    """Trainer that reports each consumed microbatch to TrainingTelemetry."""

    def __init__(self, *args, telemetry: TrainingTelemetry, **kwargs):
        self.training_telemetry = telemetry
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.training_telemetry.observe_batch(inputs)
        return super().training_step(model, inputs, num_items_in_batch)
