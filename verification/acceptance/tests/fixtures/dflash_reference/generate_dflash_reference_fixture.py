"""Writes a DFlash drafter's numerical oracle from z-lab's own MLX reference.

The Rust drafter is checked against this fixture rather than against itself. A
tiny drafter with random weights (2 layers, hidden 64, an 8-row sliding window)
runs two drafting rounds through the reference's real classes. The second round
appends context past the window and starts at a nonzero position, so the
window cut, absolute RoPE positions and context growth are all covered. The
weights keep the real checkpoints' tensor names, so the Rust loader reads these
files exactly as it reads a published drafter.

Two generations, one fixture each:

- `dflash2`: `DFlash2DraftModel` with its convolutions and candidate selector,
  shaped like `z-lab/Qwen3.8-27B-DFlash2` (every layer sliding, block not
  causal).
- `dflash`: `DFlashDraftModel`, shaped like `ornith-ai/Ornith-1.5-35B-A3B-DFlash`
  (a sliding layer and a full-attention layer, causality left undeclared, so
  the reference makes the sliding layer causal and the full layer two-way).

Run once per generation, outside the repository's toolchain, with mlx and
mlx-lm installed (mlx 0.32.4 and mlx-lm 0.31.3 produced the committed files):

    python generate_dflash_reference_fixture.py GENERATION REFERENCE_MODEL_MLX_PY OUTPUT_DIRECTORY

REFERENCE_MODEL_MLX_PY is `dflash/model_mlx.py` from https://github.com/z-lab/dflash
(MIT License, Copyright (c) 2026 Z Lab). The `dflash2` fixture came from
TensorFold's vendored copy at tag v0.3.5.1, and z-lab's main branch at
07ebd93db9f4 reproduces it byte for byte; the `dflash` fixture came from that
main branch. The output is `GENERATION_reference.safetensors`
plus `GENERATION_reference_config.json`.
"""

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx import nn

DFLASH2_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "num_target_layers": 4,
    "sliding_window": 8,
    "use_sliding_window": True,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "is_causal": False,
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 255,
        "selector_rank": 8,
        "selector_top_k": 4,
        "target_layer_ids": [1, 3],
    },
}
DFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "num_target_layers": 4,
    "sliding_window": 8,
    "use_sliding_window": True,
    "layer_types": ["sliding_attention", "full_attention"],
    "tie_word_embeddings": False,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "dflash_config": {
        "block_size": 8,
        "mask_token_id": 255,
        "target_layer_ids": [1, 3],
    },
}
CONFIGS = {"dflash2": DFLASH2_CONFIG, "dflash": DFLASH_CONFIG}
FIRST_ROUND_CONTEXT_ROWS = 12
SECOND_ROUND_CONTEXT_ROWS = 3


def load_reference(reference_path: str):
    spec = importlib.util.spec_from_file_location("z_lab_dflash_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureTarget(nn.Module):
    """Stands in for the target model: only its embedding and head are used."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def build_drafter(reference, config_document):
    dflash = config_document["dflash_config"]
    config = reference.DFlashConfig(
        hidden_size=config_document["hidden_size"], num_hidden_layers=config_document["num_hidden_layers"],
        num_attention_heads=config_document["num_attention_heads"],
        num_key_value_heads=config_document["num_key_value_heads"],
        head_dim=config_document["head_dim"], intermediate_size=config_document["intermediate_size"],
        vocab_size=config_document["vocab_size"], rms_norm_eps=config_document["rms_norm_eps"],
        rope_theta=config_document["rope_parameters"]["rope_theta"],
        max_position_embeddings=config_document["max_position_embeddings"], block_size=dflash["block_size"],
        target_layer_ids=tuple(dflash["target_layer_ids"]),
        num_target_layers=config_document["num_target_layers"],
        mask_token_id=dflash["mask_token_id"], rope_scaling=config_document["rope_parameters"],
        layer_types=tuple(config_document["layer_types"]), sliding_window=config_document["sliding_window"],
        conv_kernel_size=dflash.get("conv_kernel_size", 0), conv_group_size=dflash.get("conv_group_size", 0),
        selector_rank=dflash.get("selector_rank", 0), selector_top_k=dflash.get("selector_top_k", 0),
        is_causal=config_document.get("is_causal"),
    )
    if "DFlash2DraftModel" in config_document["architectures"]:
        drafter = reference.DFlash2DraftModel(config)
    else:
        drafter = reference.DFlashDraftModel(config)
    randomize_parameters(drafter)
    target = FixtureTarget(config_document["vocab_size"], config_document["hidden_size"])
    randomize_parameters(target)
    drafter.bind(target)
    return drafter, target


def randomize_parameters(module: nn.Module) -> None:
    """Every parameter random, norms near one, so no path hides behind a zero."""

    from mlx.utils import tree_flatten, tree_unflatten

    replaced = []
    for name, value in tree_flatten(module.parameters()):
        if name.endswith("norm.weight") or "layernorm" in name or name == "hidden_norm.weight":
            replaced.append((name, 1.0 + 0.1 * mx.random.normal(value.shape)))
        else:
            replaced.append((name, 0.2 * mx.random.normal(value.shape)))
    module.update(tree_unflatten(replaced))


def run_round(drafter, cache, anchor_id: int, context_rows, round_name: str, tensors: dict) -> None:
    mask_id = drafter.config.mask_token_id
    block = mx.array([[anchor_id] + [mask_id] * (drafter.config.block_size - 1)])
    position_before = cache[0].offset
    hidden = drafter.hidden_states(block, context_rows, cache, logits_start=1)
    logits = drafter.compute_logits(hidden)
    mx.eval(hidden, logits)
    tensors[f"oracle.{round_name}.block_ids"] = block.astype(mx.int32)
    tensors[f"oracle.{round_name}.context_rows"] = context_rows
    tensors[f"oracle.{round_name}.context_position_before"] = mx.array([position_before], dtype=mx.int32)
    tensors[f"oracle.{round_name}.final_hidden"] = hidden
    tensors[f"oracle.{round_name}.logits"] = logits
    if hasattr(drafter, "candidate_selector"):
        path, candidates, _ = drafter.candidate_selector.select(hidden, logits, block[:, 0], 0.0)
        mx.eval(path, candidates)
        tensors[f"oracle.{round_name}.greedy_path"] = path.astype(mx.int32)
        tensors[f"oracle.{round_name}.candidates"] = candidates.astype(mx.int32)


def checkpoint_tensors(drafter, target) -> dict:
    """The drafter's weights under the real checkpoint's names, plus the target rows it reads."""

    from mlx.utils import tree_flatten

    tensors = {}
    for name, value in tree_flatten(drafter.parameters()):
        if name.startswith(("embed_tokens", "lm_head")):
            continue
        if name.startswith("candidate_selector.") and name.endswith("_codebook.weight"):
            name = name[: -len(".weight")]
        tensors[name] = value
    tensors["target.embed_tokens"] = target.embed_tokens.weight
    tensors["target.lm_head"] = target.lm_head.weight
    return tensors


def main() -> int:
    generation, reference_path, output_directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    config_document = CONFIGS[generation]
    mx.random.seed(20261115)
    reference = load_reference(reference_path)
    drafter, target = build_drafter(reference, config_document)
    tensors = checkpoint_tensors(drafter, target)
    tap_width = len(config_document["dflash_config"]["target_layer_ids"]) * config_document["hidden_size"]
    cache = drafter.make_cache()
    first_context = mx.random.normal((1, FIRST_ROUND_CONTEXT_ROWS, tap_width))
    run_round(drafter, cache, 17, first_context, "first_round", tensors)
    second_context = mx.random.normal((1, SECOND_ROUND_CONTEXT_ROWS, tap_width))
    run_round(drafter, cache, 42, second_context, "second_round", tensors)
    output_directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_directory / f"{generation}_reference.safetensors"), tensors)
    (output_directory / f"{generation}_reference_config.json").write_text(
        json.dumps(config_document, indent=1) + "\n"
    )
    print(f"wrote {len(tensors)} tensors to {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
