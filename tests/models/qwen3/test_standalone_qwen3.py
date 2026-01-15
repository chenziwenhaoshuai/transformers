import torch

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.standalone import export_qwen3_to_torchscript


def test_qwen3_torchscript_export_matches_logits(tmp_path):
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

    output_path = tmp_path / "qwen3_ts.pt"
    export_qwen3_to_torchscript(model, str(output_path), example_inputs=(input_ids, attention_mask))

    ts_model = torch.jit.load(str(output_path))
    with torch.no_grad():
        traced_logits = ts_model(input_ids, attention_mask)

    torch.testing.assert_close(traced_logits, expected, rtol=1e-5, atol=1e-5)
