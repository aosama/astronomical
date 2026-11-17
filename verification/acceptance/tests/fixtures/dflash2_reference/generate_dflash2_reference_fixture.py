"""Writes the DFlash2 drafter's numerical oracle from z-lab's own MLX reference.

The Rust drafter is checked against this fixture rather than against itself. A
tiny drafter with random weights (2 layers, hidden 64, an 8-row sliding window)
runs two drafting rounds through the reference's real classes. The second round
appends context past the window and starts at a nonzero position, so the
window cut, absolute RoPE positions and context growth are all covered. The
weights keep the real checkpoint's tensor names, so the Rust loader reads this
file exactly as it reads `z-lab/Qwen3.8-27B-DFlash2`.

Run once, outside the repository's toolchain, with mlx and mlx-lm installed
(mlx 0.32.4 and mlx-lm 0.31.3 produced the committed file):

    python generate_dflash2_reference_fixture.py REFERENCE_MODEL_MLX_PY OUTPUT_DIRECTORY

REFERENCE_MODEL_MLX_PY is `dflash/model_mlx.py` from https://github.com/z-lab/dflash
(MIT License, Copyright (c) 2026 Z Lab), for example TensorFold's vendored copy at
tag v0.3.5.1. The output is `dflash2_reference.safetensors` plus
`dflash2_reference_config.json`.
"""

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx import nn

CONFIG = {
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


def build_drafter(reference):
    dflash = CONFIG["dflash_config"]
    config = reference.DFlashConfig(
        hidden_size=CONFIG["hidden_size"], num_hidden_layers=CONFIG["num_hidden_layers"],
        num_attention_heads=CONFIG["num_attention_heads"], num_key_value_heads=CONFIG["num_key_value_heads"],
        head_dim=CONFIG["head_dim"], intermediate_size=CONFIG["intermediate_size"],
        vocab_size=CONFIG["vocab_size"], rms_norm_eps=CONFIG["rms_norm_eps"],
        rope_theta=CONFIG["rope_parameters"]["rope_theta"],
        max_position_embeddings=CONFIG["max_position_embeddings"], block_size=dflash["block_size"],
        target_layer_ids=tuple(dflash["target_layer_ids"]), num_target_layers=CONFIG["num_target_layers"],
        mask_token_id=dflash["mask_token_id"], rope_scaling=CONFIG["rope_parameters"],
        layer_types=tuple(CONFIG["layer_types"]), sliding_window=CONFIG["sliding_window"],
        conv_kernel_size=dflash["conv_kernel_size"], conv_group_size=dflash["conv_group_size"],
        selector_rank=dflash["selector_rank"], selector_top_k=dflash["selector_top_k"],
        is_causal=CONFIG["is_causal"],
    )
    drafter = reference.DFlash2DraftModel(config)
    randomize_parameters(drafter)
    target = FixtureTarget(CONFIG["vocab_size"], CONFIG["hidden_size"])
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
    mask_id = CONFIG["dflash_config"]["mask_token_id"]
    block = mx.array([[anchor_id] + [mask_id] * (CONFIG["dflash_config"]["block_size"] - 1)])
    position_before = cache[0].offset
    hidden = drafter.hidden_states(block, context_rows, cache, logits_start=1)
    logits = drafter.compute_logits(hidden)
    path, candidates, _ = drafter.candidate_selector.select(hidden, logits, block[:, 0], 0.0)
    mx.eval(hidden, logits, path, candidates)
    tensors[f"oracle.{round_name}.block_ids"] = block.astype(mx.int32)
    tensors[f"oracle.{round_name}.context_rows"] = context_rows
    tensors[f"oracle.{round_name}.context_position_before"] = mx.array([position_before], dtype=mx.int32)
    tensors[f"oracle.{round_name}.final_hidden"] = hidden
    tensors[f"oracle.{round_name}.logits"] = logits
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
    reference_path, output_directory = sys.argv[1], Path(sys.argv[2])
    mx.random.seed(20261115)
    reference = load_reference(reference_path)
    drafter, target = build_drafter(reference)
    tensors = checkpoint_tensors(drafter, target)
    tap_width = len(CONFIG["dflash_config"]["target_layer_ids"]) * CONFIG["hidden_size"]
    cache = drafter.make_cache()
    first_context = mx.random.normal((1, FIRST_ROUND_CONTEXT_ROWS, tap_width))
    run_round(drafter, cache, 17, first_context, "first_round", tensors)
    second_context = mx.random.normal((1, SECOND_ROUND_CONTEXT_ROWS, tap_width))
    run_round(drafter, cache, 42, second_context, "second_round", tensors)
    output_directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_directory / "dflash2_reference.safetensors"), tensors)
    (output_directory / "dflash2_reference_config.json").write_text(json.dumps(CONFIG, indent=1) + "\n")
    print(f"wrote {len(tensors)} tensors to {output_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
