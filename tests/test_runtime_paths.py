import asyncio
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from datasets import Dataset

import api
from benchmark import run_benchmark
from lightning_data import StressDataModule
from lightning_module import StressLightningModule
from models import get_model


class DummyStaticModel(torch.nn.Module):
    def __init__(self, input_dim_sequence: int = 3, input_dim_static: int = 2):
        super().__init__()
        self.input_dim_sequence = input_dim_sequence
        self.input_dim_static = input_dim_static
        self.linear = torch.nn.Linear(input_dim_sequence + input_dim_static, 1)

    def forward(self, seq, static=None):
        pooled = seq.mean(dim=1)
        if self.input_dim_static > 0:
            if static is None:
                static = torch.zeros(seq.shape[0], self.input_dim_static, device=seq.device)
            pooled = torch.cat([pooled, static], dim=1)
        return self.linear(pooled)


class RuntimePathTests(unittest.TestCase):
    def test_unpack_supports_dict_and_tuple_batches(self):
        module = StressLightningModule(
            DummyStaticModel(),
            {"training_config": {}, "model_config": {}},
            None,
        )
        dict_batch = {
            "sequence": torch.zeros(1, 4, 3),
            "static": torch.ones(1, 2),
            "label": torch.tensor([1.0]),
        }
        seq, static, labels = module._unpack(dict_batch)
        self.assertEqual(tuple(seq.shape), (1, 4, 3))
        self.assertEqual(tuple(static.shape), (1, 2))
        self.assertEqual(tuple(labels.shape), (1,))

        tuple_batch = (
            torch.zeros(1, 4, 3),
            torch.ones(1, 2),
            torch.tensor([1.0]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        seq, static, labels = module._unpack(tuple_batch)
        self.assertEqual(tuple(seq.shape), (1, 4, 3))
        self.assertEqual(tuple(static.shape), (1, 2))
        self.assertEqual(tuple(labels.shape), (1,))

    def test_datamodule_arrow_batches_feed_lightning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            sample_dict = {
                "sequence": [
                    [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]],
                    [[0.2, 1.2, 2.2], [0.3, 1.3, 2.3]],
                ],
                "static": [[0.0, 0.5], [0.2, 0.7]],
                "label": [0.0, 1.0],
                "subject_id": [1, 2],
                "window_index": [0, 1],
            }
            for split in ("train", "val", "test"):
                Dataset.from_dict(sample_dict).save_to_disk(str(base / split))

            config = {
                "hf_path": str(base),
                "training_config": {"batch_size": 1},
                "processing": {"dataloader_num_workers": 0},
                "model_config": {},
            }
            datamodule = StressDataModule({}, {}, config)
            datamodule.setup()
            batch = next(iter(datamodule.train_dataloader()))
            self.assertIsInstance(batch, dict)
            self.assertEqual(sorted(batch.keys()), ["label", "sequence", "static"])

            module = StressLightningModule(DummyStaticModel(), config, None)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"You are trying to `self\.log\(\)` but the `self\.trainer` reference is not registered",
                    module=r"lightning\.pytorch\.core\.module",
                )
                loss = module.training_step(batch, 0)
            self.assertTrue(torch.is_tensor(loss))

    def test_api_initialization_and_prediction_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = {
                "features_to_use": {
                    "chest": ["ECG", "EDA", "ACC", "EMG"],
                    "wrist": ["EDA", "BVP", "ACC", "TEMP"],
                },
                "static_features_to_use": [],
                "model_config": {
                    "type": "LSTM",
                    "lstm_layers": [8],
                    "dropout": 0.0,
                    "bidirectional": False,
                },
                "training_config": {},
                "save_paths": {"models": str(base / "models")},
            }
            config_path = base / "config.json"
            model_dir = base / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            model = get_model(config, input_dim_sequence=8, input_dim_static=0)
            lightning_module = StressLightningModule(model, config, None)
            ckpt_path = model_dir / "dummy.ckpt"
            torch.save({"state_dict": lightning_module.state_dict()}, ckpt_path)

            api.initialize_model_state(config_path=str(config_path), model_dir=str(model_dir))
            health = api.health()
            self.assertTrue(health["model_loaded"])
            self.assertIn("cuda_available", health)
            self.assertIn("precision", health)

            response = asyncio.run(
                api.predict(
                    api.InferenceRequest(
                        sequence=[[0.0] * 8 for _ in range(16)],
                    )
                )
            )
            self.assertIsInstance(response.stress_probability, float)

    def test_benchmark_cpu_smoke(self):
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1),
        )
        run_benchmark(model, input_shape=(1, 4, 8), num_iters=2)


if __name__ == "__main__":
    unittest.main()
