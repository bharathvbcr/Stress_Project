import unittest


class MlxConvertTests(unittest.TestCase):
    def test_lstm_checkpoint_roundtrip(self):
        try:
            import mlx.core as mx
            import numpy as np
            import torch
            from models import StressLSTM
            from mlx_convert import convert_checkpoint_to_mlx
            from mlx_models import get_mlx_model, load_mlx_checkpoint
        except ImportError:
            self.skipTest("torch/mlx not installed")

        import os
        import tempfile

        config = {
            "model_config": {
                "type": "LSTM",
                "lstm_layers": [12],
                "dropout": 0.0,
                "bidirectional": True,
            }
        }
        pt = StressLSTM(4, 2, config["model_config"])
        seq = torch.randn(2, 24, 4)
        static = torch.randn(2, 2)
        with torch.no_grad():
            ref = pt(seq, static)

        fd, ckpt = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        try:
            torch.save({"state_dict": pt.state_dict()}, ckpt)
            out = ckpt.replace(".ckpt", "_mlx.npz")
            report = convert_checkpoint_to_mlx(
                config,
                ckpt,
                out,
                input_dim_sequence=4,
                input_dim_static=2,
                validate=False,
            )
            mlx_model = get_mlx_model(config, 4, 2)
            load_mlx_checkpoint(mlx_model, out)
            out_m = mlx_model(mx.array(seq.numpy()), mx.array(static.numpy()))
            diff = float(np.max(np.abs(ref.numpy() - np.array(out_m))))
            self.assertLess(diff, 1e-3)
            self.assertEqual(report["converted"], "full_model")
        finally:
            for path in (ckpt, ckpt.replace(".ckpt", "_mlx.npz")):
                if os.path.exists(path):
                    os.remove(path)

    def test_timesfm_classifier_keys_present(self):
        from mlx_models import MlxTimesFMClassifier

        head = MlxTimesFMClassifier(fc_in=1282, head_hidden_dim=64, dropout=0.1)
        params = head.parameters()
        self.assertIn("norm", params)
        self.assertIn("lin1", params)

    def test_cnn_lstm_checkpoint_roundtrip(self):
        try:
            import mlx.core as mx
            import numpy as np
            import torch
            from models import StressCNNLSTM
            from mlx_convert import convert_checkpoint_to_mlx
            from mlx_models import get_mlx_model, load_mlx_checkpoint
        except ImportError:
            self.skipTest("torch/mlx not installed")

        import os
        import tempfile

        config = {
            "model_config": {
                "type": "CNN-LSTM",
                "cnn_filters": [16, 32],
                "cnn_kernels": [3, 3],
                "lstm_layers": [16],
                "dropout": 0.0,
                "attn_heads": 4,
                "bidirectional": True,
            }
        }
        pt = StressCNNLSTM(4, 2, config["model_config"])
        pt.eval()
        seq = torch.randn(2, 24, 4)
        static = torch.randn(2, 2)
        with torch.no_grad():
            ref = pt(seq, static)

        fd, ckpt = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        try:
            torch.save({"state_dict": pt.state_dict()}, ckpt)
            out = ckpt.replace(".ckpt", "_mlx.npz")
            convert_checkpoint_to_mlx(
                config,
                ckpt,
                out,
                input_dim_sequence=4,
                input_dim_static=2,
                validate=False,
            )
            mlx_model = get_mlx_model(config, 4, 2)
            load_mlx_checkpoint(mlx_model, out)
            out_m = mlx_model(mx.array(seq.numpy()), mx.array(static.numpy()))
            diff = float(np.max(np.abs(ref.numpy() - np.array(out_m))))
            self.assertLess(diff, 0.05)
        finally:
            for path in (ckpt, ckpt.replace(".ckpt", "_mlx.npz")):
                if os.path.exists(path):
                    os.remove(path)

    def test_transformer_checkpoint_roundtrip(self):
        try:
            import mlx.core as mx
            import numpy as np
            import torch
            from models import StressTransformer
            from mlx_convert import convert_checkpoint_to_mlx
            from mlx_models import get_mlx_model, load_mlx_checkpoint
        except ImportError:
            self.skipTest("torch/mlx not installed")

        import os
        import tempfile

        config = {
            "model_config": {
                "type": "TRANSFORMER",
                "transformer_dim": 32,
                "transformer_heads": 4,
                "transformer_layers": 2,
                "transformer_ff_dim": 64,
                "dropout": 0.0,
            }
        }
        pt = StressTransformer(4, 2, config["model_config"])
        pt.eval()
        seq = torch.randn(2, 24, 4)
        static = torch.randn(2, 2)
        with torch.no_grad():
            ref = pt(seq, static)

        fd, ckpt = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        try:
            torch.save({"state_dict": pt.state_dict()}, ckpt)
            out = ckpt.replace(".ckpt", "_mlx.npz")
            convert_checkpoint_to_mlx(
                config,
                ckpt,
                out,
                input_dim_sequence=4,
                input_dim_static=2,
                validate=False,
            )
            mlx_model = get_mlx_model(config, 4, 2)
            load_mlx_checkpoint(mlx_model, out)
            out_m = mlx_model(mx.array(seq.numpy()), mx.array(static.numpy()))
            diff = float(np.max(np.abs(ref.numpy() - np.array(out_m))))
            self.assertLess(diff, 0.05)
        finally:
            for path in (ckpt, ckpt.replace(".ckpt", "_mlx.npz")):
                if os.path.exists(path):
                    os.remove(path)

    def test_patchtst_checkpoint_roundtrip(self):
        try:
            import mlx.core as mx
            import numpy as np
            import torch
            from models import StressPatchTST
            from mlx_convert import convert_checkpoint_to_mlx
            from mlx_models import get_mlx_model, load_mlx_checkpoint
        except ImportError:
            self.skipTest("torch/mlx not installed")

        import os
        import tempfile

        config = {
            "model_config": {
                "type": "PATCHTST",
                "patch_len": 8,
                "patch_stride": 4,
                "patchtst_dim": 32,
                "patchtst_heads": 4,
                "patchtst_layers": 2,
                "patchtst_ff_dim": 64,
                "dropout": 0.0,
            }
        }
        pt = StressPatchTST(4, 2, config["model_config"])
        pt.eval()
        seq = torch.randn(2, 24, 4)
        static = torch.randn(2, 2)
        with torch.no_grad():
            ref = pt(seq, static)

        fd, ckpt = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        try:
            torch.save({"state_dict": pt.state_dict()}, ckpt)
            out = ckpt.replace(".ckpt", "_mlx.npz")
            convert_checkpoint_to_mlx(
                config,
                ckpt,
                out,
                input_dim_sequence=4,
                input_dim_static=2,
                validate=False,
            )
            mlx_model = get_mlx_model(config, 4, 2)
            load_mlx_checkpoint(mlx_model, out)
            out_m = mlx_model(mx.array(seq.numpy()), mx.array(static.numpy()))
            diff = float(np.max(np.abs(ref.numpy() - np.array(out_m))))
            self.assertLess(diff, 0.05)
        finally:
            for path in (ckpt, ckpt.replace(".ckpt", "_mlx.npz")):
                if os.path.exists(path):
                    os.remove(path)

    def test_positional_encoding_saved_in_checkpoint(self):
        try:
            import mlx.core as mx
            import tempfile
            import os
            from mlx_models import MlxPositionalEncoding, save_mlx_checkpoint, load_mlx_checkpoint
        except ImportError:
            self.skipTest("mlx not installed")

        enc = MlxPositionalEncoding(16, 0.0, 32)
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            save_mlx_checkpoint(enc, path)
            data = mx.load(path)
            self.assertIn("pe", data)
            enc2 = MlxPositionalEncoding(16, 0.0, 32)
            load_mlx_checkpoint(enc2, path)
            diff = float(mx.max(mx.abs(enc.pe - enc2.pe)))
            self.assertLess(diff, 1e-6)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_timesfm_head_weight_roundtrip(self):
        try:
            import torch
            from mlx_convert import convert_timesfm_classifier, mlx_head_state_to_torch_classifier
            from mlx_models import MlxTimesFMClassifier
        except ImportError:
            self.skipTest("torch/mlx not installed")

        head = MlxTimesFMClassifier(fc_in=32, head_hidden_dim=8, dropout=0.0)
        torch_sd = {
            "classifier.0.weight": torch.randn(32),
            "classifier.0.bias": torch.randn(32),
            "classifier.1.weight": torch.randn(16, 32),
            "classifier.1.bias": torch.randn(16),
            "classifier.4.weight": torch.randn(8, 16),
            "classifier.4.bias": torch.randn(8),
            "classifier.7.weight": torch.randn(1, 8),
            "classifier.7.bias": torch.randn(1),
        }
        convert_timesfm_classifier(head, torch_sd)
        recovered = mlx_head_state_to_torch_classifier(head)
        for key, tensor in torch_sd.items():
            diff = float(torch.max(torch.abs(tensor - recovered[key])))
            self.assertLess(diff, 1e-5, msg=key)


if __name__ == "__main__":
    unittest.main()
