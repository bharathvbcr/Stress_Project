import asyncio
import json
import logging
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from datasets import Dataset

import api
import evaluate_checkpoint
from benchmark import run_benchmark
from lightning_data import StressDataModule
from lightning_module import StressLightningModule
from models import get_model
from timesfm_wrapper import TimesFMEmbeddingExtractor
from training import (
    BatchTimeLoggingCallback,
    NumericalStabilityCallback,
    build_training_profiler,
    get_standard_callbacks,
    log_gpu_diagnostics,
    training_stage,
)
from utils import get_runtime_capabilities


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
    def test_runtime_capabilities_include_mlx_fields(self):
        caps = get_runtime_capabilities()
        self.assertIn("mlx_available", caps)
        self.assertIn("mlx_version", caps)
        self.assertIn("backend_request", caps)

    def test_resolve_training_backend_prefers_mlx_on_apple_for_lstm(self):
        from mlx_backend import resolve_training_backend

        backend = resolve_training_backend(
            "LSTM",
            mlx_available=True,
            cuda_available=False,
            mps_available=True,
            requested="auto",
        )
        self.assertEqual(backend, "mlx")

    def test_mlx_cnn_lstm_forward_smoke(self):
        try:
            from mlx_models import get_mlx_model
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX not installed")

        config = {
            "model_config": {
                "type": "CNN-LSTM",
                "cnn_filters": [8, 16],
                "cnn_kernels": [5, 3],
                "cnn_stride": 1,
                "cnn_padding": "same",
                "lstm_layers": [16],
                "bidirectional": True,
                "attn_heads": 4,
                "dropout": 0.1,
            }
        }
        model = get_mlx_model(config, input_dim_sequence=4, input_dim_static=2)
        seq = mx.random.normal((2, 32, 4))
        static = mx.random.normal((2, 2))
        out = model(seq, static)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_mlx_transformer_forward_smoke(self):
        try:
            from mlx_models import get_mlx_model
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX not installed")

        config = {
            "model_config": {
                "type": "TRANSFORMER",
                "transformer_dim": 32,
                "transformer_heads": 4,
                "transformer_layers": 1,
                "transformer_ff_dim": 64,
                "dropout": 0.1,
            }
        }
        model = get_mlx_model(config, input_dim_sequence=4, input_dim_static=2)
        out = model(mx.random.normal((2, 16, 4)), mx.random.normal((2, 2)))
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_resolve_training_backend_prefers_mlx_for_cnn_lstm(self):
        from mlx_backend import resolve_training_backend

        backend = resolve_training_backend(
            "CNN-LSTM",
            mlx_available=True,
            cuda_available=False,
            mps_available=True,
            requested="auto",
        )
        self.assertEqual(backend, "mlx")

    def test_mlx_lstm_forward_smoke(self):
        try:
            from mlx_models import get_mlx_model
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX not installed")

        config = {
            "model_config": {
                "type": "LSTM",
                "lstm_layers": [8],
                "dropout": 0.0,
                "bidirectional": False,
            }
        }
        model = get_mlx_model(config, input_dim_sequence=4, input_dim_static=2)
        seq = mx.random.normal((2, 16, 4))
        static = mx.random.normal((2, 2))
        out = model(seq, static)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_runtime_capabilities_select_mps_when_cuda_absent(self):
        class FakeMPSBackend:
            @staticmethod
            def is_built():
                return True

            @staticmethod
            def is_available():
                return True

        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.object(torch.backends, "mps", FakeMPSBackend, create=True), \
             mock.patch.dict(os.environ, {"STRESS_TORCH_DEVICE": "auto"}):
            caps = get_runtime_capabilities()

        self.assertEqual(caps["device_type"], "mps")
        self.assertEqual(caps["accelerator"], "mps")
        self.assertEqual(caps["devices"], 1)
        self.assertEqual(caps["precision"], "32")

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

    def test_lightning_step_detail_logs_internal_timings(self):
        config = {
            "training_config": {
                "diagnostics": {
                    "enabled": True,
                    "step_detail": True,
                    "step_detail_interval": 1,
                    "step_detail_cuda_sync_interval": 0,
                }
            },
            "model_config": {},
        }
        module = StressLightningModule(DummyStaticModel(), config, None)
        logged_metrics = []
        module.log = lambda name, *args, **kwargs: logged_metrics.append(name)

        batch = (
            torch.zeros(2, 4, 3),
            torch.ones(2, 2),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )

        with self.assertLogs("lightning_module", level="INFO") as captured:
            loss = module.training_step(batch, 0)
            module.validation_step(batch, 0)

        self.assertTrue(torch.is_tensor(loss))
        combined_logs = "\n".join(captured.output)
        self.assertIn("Train step detail", combined_logs)
        self.assertIn("Validation step detail", combined_logs)
        self.assertIn("time/train_forward_ms", logged_metrics)
        self.assertIn("time/train_loss_ms", logged_metrics)
        self.assertIn("time/val_forward_ms", logged_metrics)
        self.assertIn("time/val_loss_ms", logged_metrics)

    def test_lightning_fit_logs_backward_and_optimizer_timings(self):
        import csv
        import lightning as L
        from lightning.pytorch.loggers import CSVLogger
        from torch.utils.data import DataLoader, TensorDataset

        config = {
            "training_config": {
                "learning_rate": 0.001,
                "lr_scheduler": {"enabled": False},
                "diagnostics": {
                    "enabled": True,
                    "log_every_n_steps": 1,
                    "cuda_sync_interval": 0,
                    "memory_log_interval": 1,
                    "step_detail": True,
                    "step_detail_interval": 1,
                    "step_detail_cuda_sync_interval": 0,
                    "epoch_memory_summary": False,
                    "weight_stats": False,
                    "stability_check_interval": 1,
                },
            },
            "early_stopping": {"patience": 1, "min_delta": 0.0},
            "model_config": {},
        }
        seq = torch.randn(4, 5, 3)
        static = torch.randn(4, 2)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loader = DataLoader(
            TensorDataset(seq, static, labels, torch.arange(4), torch.arange(4)),
            batch_size=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module = StressLightningModule(DummyStaticModel(), config, None)
            trainer = L.Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                logger=CSVLogger(tmpdir, name="diag_smoke"),
                callbacks=get_standard_callbacks(config, is_cuda=False, enable_lr_monitor=False),
                enable_checkpointing=False,
                enable_model_summary=False,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
            trainer.fit(module, loader, loader)
            metrics_path = next(Path(tmpdir).glob("diag_smoke/version_*/metrics.csv"))
            with metrics_path.open(newline="", encoding="utf-8") as fh:
                columns = next(csv.reader(fh))

        self.assertIn("time/backward_sec", columns)
        self.assertIn("time/pre_optimizer_sec", columns)
        self.assertIn("time/optimizer_step_sec", columns)
        self.assertIn("time/forward_loss_sec", columns)
        self.assertIn("perf/throughput_samples_per_sec", columns)

    def test_get_model_patchtst_forward_smoke(self):
        config = {
            "model_config": {
                "type": "PATCHTST",
                "patch_len": 4,
                "patch_stride": 2,
                "patchtst_dim": 16,
                "patchtst_heads": 4,
                "patchtst_layers": 1,
                "patchtst_ff_dim": 32,
                "dropout": 0.0,
            },
            "training_config": {},
        }
        model = get_model(config, input_dim_sequence=3, input_dim_static=2)
        logits = model(torch.randn(2, 12, 3), torch.randn(2, 2))
        self.assertEqual(tuple(logits.shape), (2, 1))

    def test_timesfm_extractor_keeps_gradients_when_backbone_is_trainable(self):
        extractor = TimesFMEmbeddingExtractor(
            checkpoint="dummy",
            context_len=4,
            horizon=1,
            normalize_inputs=True,
            device=torch.device("cpu"),
        )
        extractor.embed_dim = 4

        class DummyBackboneModel(torch.nn.Module):
            p = 1

            def __init__(self, owner):
                super().__init__()
                object.__setattr__(self, "owner", owner)
                self.linear = torch.nn.Linear(1, 4)

            def forward(self, patched_x, masks):  # noqa: ARG002
                hidden = self.linear(patched_x.float())
                self.owner._captured_hidden = hidden
                return hidden

        class DummyBackbone:
            def __init__(self, owner):
                self.model = DummyBackboneModel(owner)

        extractor._backbone = DummyBackbone(extractor)
        extractor.backbone_module = extractor._backbone.model
        extractor._hook_ok = True
        extractor._attach = lambda: None
        extractor.train()

        out = extractor(torch.ones(2, 4))
        self.assertTrue(out.requires_grad)

        for param in extractor._backbone.model.parameters():
            param.requires_grad_(False)
        out_frozen = extractor(torch.ones(2, 4))
        self.assertFalse(out_frozen.requires_grad)

    def test_timesfm_extractor_registers_backbone_in_state_dict(self):
        extractor = TimesFMEmbeddingExtractor(
            checkpoint="dummy",
            context_len=4,
            horizon=1,
            normalize_inputs=True,
            device=torch.device("cpu"),
        )

        class DummyBackboneModel(torch.nn.Module):
            p = 1

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 4)

            def forward(self, patched_x, masks):  # noqa: ARG002
                return self.linear(patched_x.float())

        extractor.backbone_module = DummyBackboneModel()
        keys = list(extractor.state_dict().keys())
        self.assertTrue(
            any(k.startswith("backbone_module.") for k in keys),
            msg=f"backbone weights missing from state_dict: {keys}",
        )

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
            self.assertGreaterEqual(response.latency_ms, 0.0)

    def test_benchmark_cpu_smoke(self):
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1),
        )
        run_benchmark(model, input_shape=(1, 4, 8), num_iters=2)

    def test_training_diagnostics_callbacks_are_configurable(self):
        config = {
            "training_config": {
                "diagnostics": {
                    "enabled": True,
                    "log_every_n_steps": 2,
                    "cuda_sync_interval": 0,
                    "memory_log_interval": 3,
                    "epoch_memory_summary": False,
                    "weight_stats": False,
                    "stability_check_interval": 7,
                }
            },
            "early_stopping": {"patience": 1, "min_delta": 0.0},
        }

        callbacks = get_standard_callbacks(config, is_cuda=False, enable_lr_monitor=False)

        batch_callback = next(c for c in callbacks if isinstance(c, BatchTimeLoggingCallback))
        stability_callback = next(c for c in callbacks if isinstance(c, NumericalStabilityCallback))
        self.assertEqual(batch_callback.log_interval, 2)
        self.assertEqual(batch_callback.cuda_sync_interval, 0)
        self.assertEqual(batch_callback.memory_log_interval, 3)
        self.assertFalse(batch_callback.epoch_memory_summary)
        self.assertEqual(stability_callback.check_every_n_steps, 7)

    def test_batch_timer_resets_around_epoch_and_validation_boundaries(self):
        callback = BatchTimeLoggingCallback(log_interval=1, cuda_sync_interval=0)
        trainer = mock.Mock(current_epoch=0, global_step=0, num_training_batches=2)
        module = mock.Mock(device=torch.device("cpu"))
        module.log = mock.Mock()

        callback.batch_start_time = 1.0
        callback.last_load_time = 99.0
        callback.on_train_epoch_start(trainer, module)
        self.assertIsNone(callback.batch_start_time)
        self.assertEqual(callback.last_load_time, 0.0)

        batch = {
            "sequence": torch.zeros(2, 4, 3),
            "static": torch.zeros(2, 2),
            "label": torch.tensor([0.0, 1.0]),
        }
        callback.on_train_batch_start(trainer, module, batch, 0)
        callback.on_train_batch_end(trainer, module, {"loss": torch.tensor(0.1)}, batch, 0)
        self.assertEqual(callback.epoch_load_time, 0.0)
        self.assertIsNotNone(callback.batch_start_time)

        callback.on_validation_epoch_start(trainer, module)
        self.assertIsNone(callback.batch_start_time)
        self.assertEqual(callback.last_load_time, 0.0)

    def test_training_profiler_env_gate_uses_profiler_results_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "STRESS_PROFILE_TRAINING": "1",
                    "STRESS_PROFILER_DIR": tmpdir,
                },
            ):
                profiler = build_training_profiler(
                    {"training_config": {"diagnostics": {"profiler": False}}},
                    run_name="unit_profile",
                )

            self.assertIsNotNone(profiler)
            self.assertTrue(Path(tmpdir).exists())

    def test_training_stage_and_gpu_diagnostics_log_without_cuda(self):
        training_logger = logging.getLogger("training")

        with self.assertLogs("training", level="INFO") as captured:
            with training_stage(training_logger, "unit_stage", torch.device("cpu"), step="smoke"):
                pass
            log_gpu_diagnostics(training_logger, torch.device("cpu"))

        combined_logs = "\n".join(captured.output)
        self.assertIn("[Training Stage] START unit_stage", combined_logs)
        self.assertIn("[Training Stage] END unit_stage", combined_logs)
        self.assertIn("CUDA is not active", combined_logs)

    def test_api_prefers_lowest_loss_full_phase2_checkpoint_and_matching_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            model_dir = base / "models"
            results_dir = base / "results"
            model_dir.mkdir()
            results_dir.mkdir()

            old = model_dir / "best_model_phase2-epoch=08-val_loss=0.3564.ckpt"
            best = model_dir / "best_model_phase2-epoch=01-val_loss=0.0573.ckpt"
            latest = model_dir / "best_model_phase2-epoch=02-val_loss=0.0621.ckpt"
            for path in (old, best, latest):
                path.write_bytes(b"ckpt")
            os.utime(old, (1, 1))
            os.utime(best, (2, 2))
            os.utime(latest, (3, 3))

            with mock.patch("api.os.path.getsize", return_value=925_000_000):
                selected = api._select_best_checkpoint(str(model_dir))
            self.assertEqual(Path(selected).name, best.name)

            (results_dir / "eval_pass_best.json").write_text(
                json.dumps(
                    {
                        "checkpoint": best.name,
                        "test_calibrated_threshold": {"threshold": 0.231},
                    }
                ),
                encoding="utf-8",
            )
            threshold, source = api._resolve_prediction_threshold(
                {"save_paths": {"results": str(results_dir)}},
                str(best),
            )
            self.assertAlmostEqual(threshold, 0.231)
            self.assertIn("eval_pass_best.json", source)

    def test_evaluate_checkpoint_writes_api_compatible_calibration_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            model_dir = base / "models"
            results_dir = base / "results"
            model_dir.mkdir()

            head_only = model_dir / "best_model_phase2-epoch=04-val_loss=0.0100.ckpt"
            best_full = model_dir / "best_model_phase2-epoch=02-val_loss=0.0700.ckpt"
            other_full = model_dir / "best_model_phase2-epoch=03-val_loss=0.0900.ckpt"
            head_only.write_bytes(b"x" * 128)
            best_full.write_bytes(b"x" * 256)
            other_full.write_bytes(b"x" * 256)

            selected = evaluate_checkpoint.select_best_checkpoint(
                model_dir,
                min_full_size_bytes=200,
            )
            self.assertEqual(selected.name, best_full.name)

            payload = evaluate_checkpoint.build_eval_payload(
                checkpoint_path=selected,
                val_probs=[0.1, 0.2, 0.8, 0.9],
                val_labels=[0, 0, 1, 1],
                test_probs=[0.1, 0.4, 0.6, 0.9],
                test_labels=[0, 1, 1, 1],
                best_threshold=0.35,
                best_val_f1=0.8,
                generated_at="2026-05-23T00:00:00+00:00",
            )
            artifact_path = evaluate_checkpoint.write_eval_payload(
                payload,
                results_dir / "eval_pass_unit.json",
            )

            threshold, source = api._resolve_prediction_threshold(
                {"save_paths": {"results": str(results_dir)}},
                str(selected),
            )
            self.assertTrue(artifact_path.exists())
            self.assertAlmostEqual(threshold, 0.35)
            self.assertIn("eval_pass_unit.json", source)

    def test_cuda_python_capabilities_presence(self):
        from utils import get_runtime_capabilities
        caps = get_runtime_capabilities()
        self.assertIn("cuda_python_available", caps)
        self.assertIn("cuda_python_version", caps)
        self.assertIn("cuda_python_package_version", caps)
        if caps["cuda_available"] and caps["cuda_python_available"]:
            from cuda import bindings
            from cuda.bindings import driver, runtime
            self.assertIsNotNone(driver)
            self.assertIsNotNone(runtime)
            self.assertIsNotNone(caps["cuda_python_version"])
            self.assertIsNotNone(caps["cuda_python_package_version"])


    def test_cuda_python_1_0_features_api_smoke(self):
        # Verify that if CUDA_PYTHON_AVAILABLE is True, we can initialize and run predict
        # with green context and custom stream settings.
        # We will mock the environment variables to trigger the green context code path.
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = {
                "features_to_use": {
                    "chest": ["ECG", "EDA", "ACC", "EMG"],
                    "wrist": ["EDA", "BVP", "ACC", "TEMP"],
                },
                "static_features_to_use": ["age", "gender"],
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

            model = get_model(config, input_dim_sequence=8, input_dim_static=2)
            lightning_module = StressLightningModule(model, config, None)
            ckpt_path = model_dir / "dummy.ckpt"
            torch.save({"state_dict": lightning_module.state_dict()}, ckpt_path)

            # Mock environments to request Green Context SM partitioning (e.g. 4 SMs)
            with mock.patch.dict(os.environ, {
                "STRESS_GREEN_CONTEXT_SM_COUNT": "4",
                "STRESS_WORKER_IDX": "0"
            }):
                # Run initialization. If cuda-python is not actually present or mock-unavailable,
                # it will fall back to normal execution/primary context, which is perfectly safe.
                api.initialize_model_state(config_path=str(config_path), model_dir=str(model_dir))
                
                # Check that health endpoint reports correctly
                health = api.health()
                self.assertTrue(health["model_loaded"])
                
                # Perform a prediction request
                response = asyncio.run(
                    api.predict(
                        api.InferenceRequest(
                            sequence=[[0.0] * 8 for _ in range(16)],
                            static=[1.0, 0.0]
                        )
                    )
                )
                self.assertIsInstance(response.stress_probability, float)
                self.assertGreaterEqual(response.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
