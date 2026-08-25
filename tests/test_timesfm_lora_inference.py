import unittest


class TimesfmLoRAInferenceTests(unittest.TestCase):
    def test_restore_lora_changes_linear_output(self):
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch not installed")

        from timesfm_lora import (
            apply_lora_to_linear,
            clear_lora_hooks,
            restore_timesfm_lora_adapters,
        )

        class DummyTimesFM(nn.Module):
            def __init__(self):
                super().__init__()
                self.extractor = nn.Module()
                self.extractor._backbone = nn.Module()
                block = nn.Module()
                block.sub = nn.Linear(6, 4)
                self.extractor._backbone.model = nn.Module()
                self.extractor._backbone.model.stacked_xf = nn.ModuleList([block])
                self.lora_adapters = nn.ModuleList()
                self._lora_hook_handles = []
                self._lora_last_n_blocks = 1

            def _ensure_attached(self):
                return None

        model = DummyTimesFM()
        linear = model.extractor._backbone.model.stacked_xf[0].sub
        x = torch.randn(2, 6)
        with torch.no_grad():
            base = linear(x)

        adapter, handle = apply_lora_to_linear(linear, rank=2, alpha=4.0)
        with torch.no_grad():
            adapter.lora_b.fill_(0.1)
        hooked = linear(x)
        handle.remove()

        state = {"lora_adapters.0.lora_a": adapter.lora_a.detach().clone(), "lora_adapters.0.lora_b": adapter.lora_b.detach().clone()}
        restored = restore_timesfm_lora_adapters(model, state, last_n_blocks=1)
        self.assertTrue(restored)
        with torch.no_grad():
            out = linear(x)
        clear_lora_hooks(model)
        self.assertFalse(torch.allclose(base, out))
        self.assertFalse(torch.allclose(base, hooked))


if __name__ == "__main__":
    unittest.main()
