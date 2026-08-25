import unittest

import numpy as np


class MlxModelSmokeTests(unittest.TestCase):
    def test_all_supported_mlx_models_forward(self):
        try:
            from mlx_models import get_mlx_model
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX not installed")

        configs = [
            {"type": "LSTM", "lstm_layers": [8], "dropout": 0.0, "bidirectional": True},
            {
                "type": "PATCHTST",
                "patch_len": 8,
                "patch_stride": 4,
                "patchtst_dim": 32,
                "patchtst_heads": 4,
                "patchtst_layers": 1,
                "patchtst_ff_dim": 64,
                "dropout": 0.0,
            },
            {
                "type": "TRANSFORMER",
                "transformer_dim": 32,
                "transformer_heads": 4,
                "transformer_layers": 1,
                "transformer_ff_dim": 64,
                "dropout": 0.0,
            },
            {
                "type": "CNN-LSTM",
                "cnn_filters": [8, 16],
                "cnn_kernels": [5, 3],
                "lstm_layers": [16],
                "bidirectional": True,
                "attn_heads": 4,
                "dropout": 0.1,
            },
        ]
        seq = mx.random.normal((2, 32, 4))
        static = mx.random.normal((2, 2))
        for model_config in configs:
            model = get_mlx_model({"model_config": model_config}, 4, 2)
            out = model(seq, static)
            self.assertEqual(tuple(out.shape), (2, 1))

    def test_backend_resolution_for_cnn_lstm(self):
        from mlx_backend import resolve_training_backend

        backend = resolve_training_backend(
            "CNN-LSTM",
            mlx_available=True,
            cuda_available=False,
            mps_available=True,
            requested="auto",
        )
        self.assertEqual(backend, "mlx")

    def test_timesfm_not_on_mlx(self):
        from mlx_backend import mlx_supports_model

        self.assertFalse(mlx_supports_model("TIMESFM"))

    def test_compiled_step_keeps_dropout_random(self):
        """A compiled step must thread mx.random.state so Dropout re-randomizes.

        Without the RNG state in inputs/outputs, mx.compile freezes a single
        dropout mask for every call, silently disabling regularization.
        """
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError:
            self.skipTest("MLX not installed")

        dropout = nn.Dropout(0.5)
        state = [mx.random.state]

        def _apply(x):
            return dropout(x)

        compiled = mx.compile(_apply, inputs=state, outputs=state)
        x = mx.ones((256,))
        out1 = compiled(x)
        mx.eval(out1, state)
        out2 = compiled(x)
        mx.eval(out2, state)
        self.assertFalse(bool(mx.array_equal(out1, out2).item()))

    def test_benchmark_measures_forward_output(self):
        """measure_mlx_latency must realize the forward output, not the params."""
        try:
            import mlx.core as mx
            from mlx_models import get_mlx_model
            from mlx_benchmark import measure_mlx_latency
        except ImportError:
            self.skipTest("MLX not installed")

        cfg = {"model_config": {
            "type": "PATCHTST", "patch_len": 8, "patch_stride": 4,
            "patchtst_dim": 64, "patchtst_heads": 4, "patchtst_layers": 2,
            "patchtst_ff_dim": 128, "dropout": 0.0,
        }}
        model = get_mlx_model(cfg, 6, 3)
        mx.eval(model.parameters())
        seq = mx.random.normal((8, 128, 6))
        static = mx.random.normal((8, 3))
        stats = measure_mlx_latency(model, seq, static, num_iters=20, warmup_iters=5)
        # If the forward were never realized (the old bug) this collapses to
        # sub-millisecond scheduling time; a real forward is comfortably above.
        self.assertGreater(stats["avg"], 0.05)
        self.assertGreater(stats["throughput"], 0.0)

    def test_strided_patches_match_reference_loop(self):
        """as_strided patch extraction must equal the original slice-and-stack."""
        try:
            import mlx.core as mx
            from mlx_models import _extract_patches
        except ImportError:
            self.skipTest("MLX not installed")

        def _ref(x, patch_len, stride):
            _, _, length = x.shape
            if length < patch_len:
                x = mx.pad(x, [(0, 0), (0, 0), (patch_len - length, 0)])
                length = patch_len
            sl = [x[:, :, s : s + patch_len] for s in range(0, length - patch_len + 1, stride)]
            return mx.stack(sl, axis=2)

        for length, plen, stride in [(300, 16, 8), (64, 16, 16), (10, 16, 8)]:
            # feed a transposed (non-contiguous) view, matching the real caller
            x = mx.transpose(mx.random.normal((4, length, 6)), (0, 2, 1))
            got = _extract_patches(x, plen, stride)
            ref = _ref(x, plen, stride)
            mx.eval(got, ref)
            self.assertEqual(tuple(got.shape), tuple(ref.shape))
            self.assertTrue(bool(mx.array_equal(got, ref).item()), (length, plen, stride))

    def test_iter_mlx_batches_coverage_and_drop_last(self):
        """Preloaded batch iterator: shuffle covers all rows; drop_last trims tail."""
        try:
            import mlx.core as mx
            from mlx_training import _iter_mlx_batches
        except ImportError:
            self.skipTest("MLX not installed")

        n = 10
        labels = mx.arange(n).reshape(-1, 1)
        split = {"seq": mx.zeros((n, 4, 2)), "static": None, "labels": labels, "n": n}

        # drop_last=False keeps every row exactly once, order preserved
        seen = []
        for _, _, y in _iter_mlx_batches(split, 4, drop_last=False):
            seen.extend(np.asarray(y).reshape(-1).tolist())
        self.assertEqual(sorted(seen), list(range(n)))

        # drop_last=True drops the trailing partial batch (10 -> 8 rows)
        kept = sum(int(y.shape[0]) for _, _, y in _iter_mlx_batches(split, 4, drop_last=True))
        self.assertEqual(kept, 8)

        # shuffle still covers all rows but changes order
        shuffled = []
        for _, _, y in _iter_mlx_batches(split, 4, shuffle=True, seed=3, drop_last=False):
            shuffled.extend(np.asarray(y).reshape(-1).tolist())
        self.assertEqual(sorted(shuffled), list(range(n)))
        self.assertNotEqual(shuffled, list(range(n)))


if __name__ == "__main__":
    unittest.main()
