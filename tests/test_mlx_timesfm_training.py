import os
import unittest


class MlxTimesfmTrainingTests(unittest.TestCase):
    def test_should_use_mlx_timesfm_phase1_respects_env(self):
        from mlx_timesfm_training import should_use_mlx_timesfm_phase1

        caps = {"backend_request": "auto", "mlx_available": True}
        prev = os.environ.get("TIMESFM_MLX_PHASE1")
        try:
            os.environ["TIMESFM_MLX_PHASE1"] = "0"
            self.assertFalse(should_use_mlx_timesfm_phase1("TIMESFM", caps))
            os.environ["TIMESFM_MLX_PHASE1"] = "1"
            # resolve_timesfm_hybrid_enabled needs mlx installed
            try:
                import mlx  # noqa: F401
            except ImportError:
                self.skipTest("mlx not installed")
            self.assertTrue(should_use_mlx_timesfm_phase1("TIMESFM", caps))
            self.assertFalse(should_use_mlx_timesfm_phase1("LSTM", caps))
        finally:
            if prev is None:
                os.environ.pop("TIMESFM_MLX_PHASE1", None)
            else:
                os.environ["TIMESFM_MLX_PHASE1"] = prev


if __name__ == "__main__":
    unittest.main()
