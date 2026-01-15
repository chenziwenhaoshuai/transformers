from typing import Optional, Tuple

import torch
from torch import nn

from .modeling_qwen3 import Qwen3ForCausalLM

DEFAULT_TRACE_SEQ_LEN = 8


class Qwen3TorchScriptWrapper(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits


def export_qwen3_to_torchscript(
    model: Qwen3ForCausalLM,
    output_path: str,
    example_inputs: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
) -> str:
    """
    Export a ``Qwen3ForCausalLM`` model to a TorchScript artifact that can be loaded with PyTorch alone.
    The saved module only depends on ``torch.jit.load`` for inference.
    """
    model.eval()
    device = next(model.parameters()).device

    if example_inputs is None:
        seq_len = min(DEFAULT_TRACE_SEQ_LEN, model.config.max_position_embeddings)
        input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        example_inputs = (input_ids, attention_mask)

    input_ids, attention_mask = example_inputs
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None

    scripted_module = torch.jit.trace(Qwen3TorchScriptWrapper(model), (input_ids, attention_mask))
    scripted_module.save(output_path)
    return output_path
