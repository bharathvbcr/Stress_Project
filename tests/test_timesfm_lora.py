import unittest


class TimesfmLoRATests(unittest.TestCase):
    def test_lora_adapter_changes_output(self):
        import torch
        from torch import nn

        from timesfm_lora import LoRAAdapter, apply_lora_to_linear

        linear = nn.Linear(8, 4, bias=True)
        x = torch.randn(3, 8)
        with torch.no_grad():
            base = linear(x)
        adapter, handle = apply_lora_to_linear(linear, rank=2, alpha=4.0)
        with torch.no_grad():
            adapter.lora_b.fill_(0.05)
        out = linear(x)
        handle.remove()
        self.assertEqual(tuple(adapter.lora_a.shape), (2, 8))
        self.assertFalse(torch.allclose(base, out))
        self.assertTrue(all(not p.requires_grad for p in linear.parameters()))
        self.assertTrue(any(p.requires_grad for p in adapter.parameters()))

    def test_should_use_timesfm_lora_phase2_env(self):
        import os

        from timesfm_lora import should_use_timesfm_lora_phase2

        caps = {"backend_request": "auto", "mlx_available": True}
        prev = os.environ.get("TIMESFM_LORA_PHASE2")
        try:
            os.environ["TIMESFM_LORA_PHASE2"] = "0"
            self.assertFalse(should_use_timesfm_lora_phase2(caps))
            os.environ["TIMESFM_LORA_PHASE2"] = "1"
            self.assertTrue(should_use_timesfm_lora_phase2(caps))
        finally:
            if prev is None:
                os.environ.pop("TIMESFM_LORA_PHASE2", None)
            else:
                os.environ["TIMESFM_LORA_PHASE2"] = prev


if __name__ == "__main__":
    unittest.main()
