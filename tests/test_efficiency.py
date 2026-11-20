"""Tests for Computational Efficiency Benchmark Conformance (MyoPS-Net).

Strictly verifies:
- "Quy tắc đo hiệu suất tính toán.txt"
- Params (Total and Trainable)
- Theoretical FLOPs on tensor (1, 3, 128, 128) FP32
- Inference measurement (Batch=1)
- Training efficiency metrics extraction
- JSON schema and directory structure (outputs/runs/myopsnet/efficiency_metrics.json)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

# Ensure MyoPS-Net root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_efficiency import (
    compute_flops,
    measure_inference,
    extract_training_metrics,
    run_benchmark,
)
from network.model import MyoPSNetL


class TestMyoPSEfficiencyBenchmark(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MyoPSNetL().to(self.device).eval()

    def test_parameter_counts(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total, 119284360)
        self.assertEqual(trainable, 119284360)
        total_m = round(total / 1e6, 2)
        self.assertEqual(total_m, 119.28)

    def test_compute_flops_fp32(self):
        flops_g = compute_flops(self.model, input_size=(1, 3, 128, 128), device=self.device)
        self.assertIsInstance(flops_g, float)
        self.assertGreater(flops_g, 50.0)
        self.assertLess(flops_g, 250.0)

    def test_measure_inference(self):
        metrics = measure_inference(self.model, self.device, warmup_runs=2, measure_runs=3)
        self.assertIn("latency_ms_per_slice", metrics)
        self.assertIn("peak_vram_mb", metrics)
        self.assertGreater(metrics["latency_ms_per_slice"], 0.0)

    def test_extract_training_metrics_from_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "training_summary_seed1234.json"
            dummy_summary = {
                "avg_epoch_time_epoch_6_to_300_s": 22.35,
                "total_training_hours": 1.95,
                "peak_train_vram_gb": 8.75,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(dummy_summary, f)

            extracted = extract_training_metrics(summary_path=summary_path, seed=1234)
            self.assertEqual(extracted["time_per_epoch_s"], 22.35)
            self.assertEqual(extracted["total_time_hours"], 1.95)
            self.assertEqual(extracted["peak_vram_gb"], 8.75)

    def test_run_benchmark_and_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "outputs" / "runs" / "myopsnet" / "efficiency_metrics.json"
            report = run_benchmark(
                checkpoint_path=None,
                output_path=out_file,
                summary_path=None,
                device_name="cpu",
                seed=1234,
            )
            self.assertTrue(out_file.is_file())

            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["model_id"], "myopsnet")
            self.assertEqual(loaded["precision"]["mode"], "FP32")
            self.assertEqual(loaded["precision"]["amp"], "none")
            self.assertEqual(loaded["dataloader"]["num_workers"], 0)
            self.assertEqual(loaded["dataloader"]["pin_memory"], True)
            self.assertEqual(loaded["complexity"]["total_params_M"], 119.28)
            self.assertIn("flops_G", loaded["complexity"])
            self.assertIn("latency_ms_per_slice", loaded["inference"])
            self.assertEqual(loaded["training"]["seed"], 1234)
            self.assertEqual(loaded["training"]["max_epochs"], 300)
            self.assertEqual(loaded["training"]["batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
