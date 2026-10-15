"""Fast offline integration test for Experiment D's telemetry-enabled Trainer."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, BertConfig, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dapt_equal_update.telemetry import TokenCountingTrainer, TrainingTelemetry

EXP = ROOT / "experiments" / "dapt" / "equal_update"


class TinyMLMDataset(Dataset):
    rows = [
        ([2, 5, 6, 3, 0], [1, 1, 1, 1, 0]),
        ([2, 7, 3, 0, 0], [1, 1, 1, 0, 0]),
        ([2, 8, 9, 10, 3], [1, 1, 1, 1, 1]),
        ([2, 11, 12, 3, 0], [1, 1, 1, 1, 0]),
    ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, mask = self.rows[index]
        labels = [-100, ids[1], -100, -100, -100]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
        }


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trainer_smoke_", dir=EXP) as temporary:
        output = Path(temporary)
        telemetry_path = output / "training_telemetry.json"
        telemetry = TrainingTelemetry(
            telemetry_path,
            {"optimizer_updates": 2, "warmup_steps": 1, "peak_learning_rate": 2e-5},
        )
        model = AutoModelForMaskedLM.from_config(BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
        ))
        trainer = TokenCountingTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(output), use_cpu=True, max_steps=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                learning_rate=2e-5, warmup_steps=1, lr_scheduler_type="linear",
                logging_steps=1, save_strategy="no", report_to="none", seed=7,
            ),
            train_dataset=TinyMLMDataset(),
            telemetry=telemetry,
            callbacks=[telemetry],
        )
        trainer.train()
        report = json.loads(telemetry_path.read_text(encoding="utf8"))
        assert report["completed_optimizer_updates"] == 2
        assert len(report["tokens_by_step"]) == 2
        assert report["total_non_padding_tokens_processed"] > 0
        assert len(report["loss_lr_trajectory"]) == 2
        assert report["observed_peak_learning_rate"] == 2e-5
        assert report["observed_final_learning_rate"] == 0.0
    print("Experiment D Trainer integration smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
