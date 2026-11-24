#!/usr/bin/env python3
"""Write a tiny random Qwen3.5 mixture-of-experts checkpoint in the layout an
original Hugging Face release uses (fused experts, PyTorch conv axes,
RMSNorm gamma - 1), for converter parity checks that need no real model.

The checkpoint is a few megabytes and is not a working model: it exists so
OptiQ and the Rust converter can be run on the same mixture-of-experts
layout (fused gate-up tensors, a router, a shared expert, an untied head)
and their artifacts compared with compare_optiq_artifacts.py. The serving
engine's config validation refuses it, so it exercises the structural
method only.

Pass --tie-word-embeddings for a variant without an output head of its own.
OptiQ's uniform-reference sweep cannot match an untied `lm_head` to its bf16
source, and in a model this small the head is most of the weights, so its
measured-majority guard refuses the untied variant.

Usage: build_optiq_synthetic_moe_source.py <qwen3_5_model_dir_for_config_and_tokenizer> <output_dir>
           [--tie-word-embeddings]
"""
import json, shutil, sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5_moe

NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                 "model.norm.weight", ".q_norm.weight", ".k_norm.weight")

donor, out = Path(sys.argv[1]), Path(sys.argv[2])
ties_word_embeddings = "--tie-word-embeddings" in sys.argv[3:]
print("[synthetic-moe] 1/3 (33%) building the model skeleton, eta under 5s", flush=True)
config = json.loads((donor / "config.json").read_text())
for key in ("quantization", "quantization_config", "vision_config", "image_token_id",
            "video_token_id", "vision_start_token_id", "vision_end_token_id"):
    config.pop(key, None)
config["model_type"] = "qwen3_5_moe"
config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
config["text_config"].update(
    model_type="qwen3_5_moe_text", hidden_size=128, intermediate_size=256, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
    shared_expert_intermediate_size=64, tie_word_embeddings=ties_word_embeddings,
    mtp_num_hidden_layers=0)
config["tie_word_embeddings"] = ties_word_embeddings
if "layer_types" in config["text_config"]:
    config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:8]
mx.random.seed(7)
model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
tensors = {}
for name, value in tree_flatten(model.parameters()):
    value = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16) if value.ndim >= 2 else value.astype(mx.bfloat16)
    if name.endswith("A_log"):
        value = value.astype(mx.float32)
    tensors[name] = value
hub = {}
for name, value in tensors.items():
    hub_name = "model.language_model." + name[len("language_model.model."):] if name.startswith("language_model.model.") else name[len("language_model."):]
    if "conv1d.weight" in name and value.ndim == 3:
        value = value.moveaxis(1, 2)
    elif value.ndim == 1 and any(name.endswith(s) for s in NORM_SUFFIXES):
        value = value - 1.0
    hub[hub_name] = value
layers = config["text_config"]["num_hidden_layers"]
for layer in range(layers):
    prefix = f"model.language_model.layers.{layer}.mlp"
    gate = hub.pop(f"{prefix}.switch_mlp.gate_proj.weight")
    up = hub.pop(f"{prefix}.switch_mlp.up_proj.weight")
    hub[f"{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
    hub[f"{prefix}.experts.down_proj"] = hub.pop(f"{prefix}.switch_mlp.down_proj.weight")
print("[synthetic-moe] 2/3 (67%) materializing the weights, eta under 5s", flush=True)
mx.eval(*hub.values())
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
mx.save_safetensors(str(out / "model.safetensors"), hub, metadata={"format": "pt"})
(out / "config.json").write_text(json.dumps(config, indent=2))
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(donor / name, out / name)
print(f"[synthetic-moe] 3/3 (100%) wrote {out} ({len(hub)} tensors)", flush=True)
