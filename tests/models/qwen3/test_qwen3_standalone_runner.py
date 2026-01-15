import importlib
import sys
from pathlib import Path

import torch

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.standalone import export_qwen3_to_torchscript


def test_root_standalone_runner_uses_torch_only(tmp_path, monkeypatch):
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=128,
        rms_norm_eps=1e-5,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask).logits

    ts_path = tmp_path / "qwen3_ts.pt"
    export_qwen3_to_torchscript(model, str(ts_path), example_inputs=(input_ids, attention_mask))

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
    try:
        standalone = importlib.import_module("qwen3_standalone")
        ts_model = standalone.load_torchscript(str(ts_path))
        with torch.no_grad():
            logits = standalone.run_torchscript(ts_model, input_ids, attention_mask)
    finally:
        sys.path.pop(0)

    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)
