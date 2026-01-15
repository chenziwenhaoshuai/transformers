"""
Lightweight Torch-only loader for Qwen3 TorchScript exports.

Usage:
    python qwen3_standalone.py --model qwen3_ts.pt --input "[[1,2,3]]"
"""

import argparse
import ast
from typing import Optional

import torch


def load_torchscript(model_path: str) -> torch.jit.ScriptModule:
    return torch.jit.load(model_path)


def run_torchscript(
    model: torch.jit.ScriptModule, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return model(input_ids, attention_mask)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 TorchScript with PyTorch only.")
    parser.add_argument("--model", required=True, help="Path to TorchScript .pt file exported from Qwen3.")
    parser.add_argument(
        "--input",
        required=True,
        help='Python literal for input_ids, e.g. "[[1,2,3]]". dtype long is assumed.',
    )
    parser.add_argument(
        "--attention_mask",
        default=None,
        help='Optional Python literal for attention_mask, e.g. "[[1,1,1]]". If omitted, no mask is used.',
    )
    args = parser.parse_args()

    input_ids = torch.tensor(ast.literal_eval(args.input), dtype=torch.long)
    attention_mask = None if args.attention_mask is None else torch.tensor(ast.literal_eval(args.attention_mask))

    model = load_torchscript(args.model)
    with torch.no_grad():
        logits = run_torchscript(model, input_ids, attention_mask)
    print(logits)


if __name__ == "__main__":
    main()
