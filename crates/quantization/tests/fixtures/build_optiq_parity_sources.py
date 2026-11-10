#!/usr/bin/env python3
"""Build two bf16 parity sources from one installed MLX-quantized Qwen3.5 model.

``mlx_layout``  : the language tower dequantized to bf16 under mlx-lm names.
``hub_layout``  : the same weights spelled the way an original Hugging Face
                  release spells them (``model.language_model.*``,
                  ``model.visual.*``, PyTorch conv axes, RMSNorm ``gamma - 1``),
                  so a converter's sanitize step is exercised end to end.

Usage: build_parity_sources.py <quantized_model_dir> <output_root>
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
COPIED_SUFFIXES = (".json", ".jinja", ".txt", ".md")


def report(stage: str, done: int, total: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    remaining = elapsed / done * (total - done) if done else float("nan")
    print(f"  [{stage}] {done}/{total} ({100.0 * done / total:.0f}%) "
          f"elapsed {elapsed:.1f}s eta {remaining:.1f}s", flush=True)


def dequantized_language_tensors(source_dir: Path) -> dict:
    config = json.loads((source_dir / "config.json").read_text())
    quantization = config["quantization"]
    stored = mx.load(str(source_dir / "model.safetensors"))
    float_tensors: dict = {}
    names = sorted(stored)
    started_at, last_report = time.time(), 0.0
    for position, name in enumerate(names, 1):
        if name.endswith((".scales", ".biases")):
            continue
        tensor = stored[name]
        module_base = name[: -len(".weight")] if name.endswith(".weight") else None
        if module_base and f"{module_base}.scales" in stored:
            entry = quantization.get(module_base)
            bits = entry["bits"] if isinstance(entry, dict) else quantization["bits"]
            group = entry["group_size"] if isinstance(entry, dict) else quantization["group_size"]
            tensor = mx.dequantize(
                tensor, stored[f"{module_base}.scales"], stored[f"{module_base}.biases"],
                group_size=group, bits=bits).astype(mx.bfloat16)
        mx.eval(tensor)
        float_tensors[name] = tensor
        if time.time() - last_report >= 5.0 or position == len(names):
            report("dequantize", position, len(names), started_at)
            last_report = time.time()
    return float_tensors


def hub_spelling(name: str, tensor):
    if name.startswith("language_model.model."):
        hub_name = "model.language_model." + name[len("language_model.model."):]
    elif name.startswith("vision_tower."):
        hub_name = "model.visual." + name[len("vision_tower."):]
    else:
        raise ValueError(f"unexpected tensor root: {name}")
    if "conv1d.weight" in name and tensor.ndim == 3:
        tensor = tensor.moveaxis(1, 2)
    elif "patch_embed.proj.weight" in name and tensor.ndim == 5:
        tensor = tensor.transpose(0, 4, 1, 2, 3)
    elif tensor.ndim == 1 and any(name.endswith(suffix) for suffix in NORM_SUFFIXES):
        tensor = tensor - 1.0
    return hub_name, tensor


def write_source(target_dir: Path, tensors: dict, source_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    mx.save_safetensors(str(target_dir / "model.safetensors"), tensors, metadata={"format": "pt"})
    for entry in source_dir.iterdir():
        if entry.is_file() and entry.suffix in COPIED_SUFFIXES and entry.name != "model.safetensors.index.json":
            shutil.copy(entry, target_dir / entry.name)
    config = json.loads((target_dir / "config.json").read_text())
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    (target_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  wrote {target_dir} ({len(tensors)} tensors)", flush=True)


def main() -> None:
    source_dir, output_root = Path(sys.argv[1]), Path(sys.argv[2])
    mlx_tensors = dequantized_language_tensors(source_dir)
    write_source(output_root / "mlx_layout", mlx_tensors, source_dir)
    hub_tensors = dict(hub_spelling(name, tensor) for name, tensor in mlx_tensors.items())
    mx.eval(*hub_tensors.values())
    write_source(output_root / "hub_layout", hub_tensors, source_dir)


if __name__ == "__main__":
    main()
