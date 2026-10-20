#!/usr/bin/env python3
"""Generate the parity fixtures that pin the Rust port to the OptiQ reference.

Run with a Python environment that has ``mlx-optiq`` (the version is recorded
in every fixture) and MLX installed. The fixtures are small JSON documents
checked into ``crates/quantization/tests/fixtures/``; the Rust offline tests
read them and assert that the ported allocator, static scorer, taxonomy, and
reference encoder reproduce the Python results exactly (allocation, taxonomy,
packed codes) or within float rounding (scales, biases, KL).

Regenerate only when deliberately re-pinning to a new OptiQ release, and
review the diff: a changed fixture is a changed contract.

Progress is reported for every fixture with a percentage and an estimated
time to completion, per the repository rule for batch scripts.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _report(stage: str, done: int, total: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    remaining = (elapsed / done) * (total - done) if done else float("nan")
    percent = 100.0 * done / total if total else 100.0
    print(
        f"  [{stage}] {done}/{total} ({percent:.0f}%) elapsed {elapsed:.1f}s "
        f"eta {remaining:.1f}s",
        flush=True,
    )


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {path.name} ({path.stat().st_size} bytes)", flush=True)


def _synthetic_sensitivity_table(seed: int) -> list:
    """A small dense-plus-MoE table whose shape exercises every allocator rule:
    four blocks of full attention plus feed-forward, one MoE block with a
    router and shared experts, an embedding, a head, and one NaN module."""
    from optiq.core.sensitivity import SensitivityResult

    rng = np.random.RandomState(seed)
    results = []

    def add(name: str, params: int, scale: float) -> None:
        kl_at_two = scale * (1.0 + rng.rand())
        results.append(SensitivityResult(
            layer_name=name,
            sensitivities={2: kl_at_two, 3: kl_at_two * 0.45, 4: kl_at_two * 0.2,
                           8: kl_at_two * 0.02},
            param_count=params,
        ))

    add("model.embed_tokens", 4_000_000, 0.9)
    for block in range(4):
        prefix = f"model.layers.{block}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"{prefix}.self_attn.{projection}", 1_000_000, 0.3 + 0.1 * block)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            add(f"{prefix}.mlp.{projection}", 3_000_000, 0.8 - 0.1 * block)
    add("model.layers.4.linear_attn.in_proj_qkv", 2_000_000, 0.4)
    add("model.layers.4.linear_attn.out_proj", 1_000_000, 0.4)
    add("model.layers.4.mlp.gate", 10_000, 0.05)
    add("model.layers.4.mlp.shared_experts.up_proj", 500_000, 0.6)
    add("model.layers.4.mlp.switch_mlp.gate_proj", 64_000_000, 0.2)
    add("model.layers.4.mlp.switch_mlp.up_proj", 64_000_000, 0.2)
    add("model.layers.4.mlp.switch_mlp.down_proj", 64_000_000, 0.25)
    add("lm_head", 4_000_000, 0.9)
    # One module with a NaN measurement at the lowest width.
    nan_entry = results[7]
    nan_entry.sensitivities[2] = float("nan")
    return results


def generate_allocation_parity(output_directory: Path) -> None:
    import io
    import contextlib

    from optiq.core.optimizer import optimize_mixed_precision

    started_at = time.time()
    table = _synthetic_sensitivity_table(seed=7)
    cases = [
        {"target_bpw": 3.0, "candidate_bits": [2, 3, 4, 8], "n_floor_per_block": 2,
         "max_low_bit_run": 3, "protect_first_last": True},
        {"target_bpw": 4.5, "candidate_bits": [2, 3, 4, 8], "n_floor_per_block": 2,
         "max_low_bit_run": 3, "protect_first_last": True},
        {"target_bpw": 4.5, "candidate_bits": [4, 8], "n_floor_per_block": 0,
         "max_low_bit_run": 0, "protect_first_last": False},
        {"target_bpw": 6.0, "candidate_bits": [4, 8], "n_floor_per_block": 2,
         "max_low_bit_run": 3, "protect_first_last": True},
        {"target_bpw": 2.5, "candidate_bits": [2, 4], "n_floor_per_block": 1,
         "max_low_bit_run": 2, "protect_first_last": True},
    ]
    outputs = []
    for index, case in enumerate(cases, 1):
        with contextlib.redirect_stdout(io.StringIO()):
            result = optimize_mixed_precision(
                table,
                target_bpw=case["target_bpw"],
                candidate_bits=case["candidate_bits"],
                group_size=64,
                protect_first_last=case["protect_first_last"],
                n_floor_per_block=case["n_floor_per_block"],
                max_low_bit_run=case["max_low_bit_run"],
            )
        outputs.append({
            "constraints": case,
            "achieved_bpw": result.achieved_bpw,
            "n_high_bits": result.n_high_bits,
            "n_low_bits": result.n_low_bits,
            "allocation": {c.layer_name: c.bits for c in result.configs},
        })
        _report("allocation", index, len(cases), started_at)

    def encode_value(value: float) -> object:
        return "NaN" if isinstance(value, float) and math.isnan(value) else value

    _write(output_directory / "allocation_parity.json", {
        "optiq_version": _optiq_version(),
        "sensitivity_table": [
            {"layer_name": r.layer_name,
             "sensitivities": {str(b): encode_value(v) for b, v in r.sensitivities.items()},
             "param_count": r.param_count}
            for r in table
        ],
        "cases": outputs,
    })


def generate_taxonomy_parity(output_directory: Path) -> None:
    from optiq.core.optimizer import (
        _identify_boundary_layers, _identify_moe_protected_layers,
        _identify_output_layers, _is_attention_component, _block_index,
    )
    from optiq.core.shardwise import is_expert, is_router

    started_at = time.time()
    names = [r.layer_name for r in _synthetic_sensitivity_table(seed=7)] + [
        "model.layers.9.mlp.router", "model.layers.9.mlp.router.proj",
        "model.layers.9.mlp.experts.3.down_proj", "model.layers.9.ffn.gate.weight",
        "model.layers.9.mlp.gate.expert_bias", "mtp.fc", "model.norm",
        "model.layers.9.mlp.switch_glu.up_proj", "model.layers.9.attn.Wqkv",
    ]
    _report("taxonomy", 1, 1, started_at)
    _write(output_directory / "taxonomy_parity.json", {
        "optiq_version": _optiq_version(),
        "module_names": names,
        "block_index": {n: _block_index(n) for n in names},
        "is_attention_component": {n: _is_attention_component(n) for n in names},
        "output_layers": sorted(_identify_output_layers(names)),
        "moe_protected_layers": sorted(_identify_moe_protected_layers(names)),
        "boundary_layers_n1": sorted(_identify_boundary_layers(names, 1)),
        "shardwise_is_expert": {f"{n}.weight": is_expert(f"{n}.weight") for n in names},
        "shardwise_is_router": {f"{n}.weight": is_router(f"{n}.weight") for n in names},
    })


def generate_static_scorer_parity(output_directory: Path) -> None:
    from optiq.core.sensitivity import _structural_priority

    started_at = time.time()
    names = [r.layer_name for r in _synthetic_sensitivity_table(seed=7)]
    n_layers = 5
    _report("static scorer", 1, 1, started_at)
    _write(output_directory / "static_scorer_parity.json", {
        "optiq_version": _optiq_version(),
        "n_layers": n_layers,
        "structural_priority": {n: _structural_priority(n, n_layers) for n in names},
    })


def generate_encoder_parity(output_directory: Path) -> None:
    import mlx.core as mx
    from optiq.core.lowbit import quantize_best
    from optiq.core.sensitivity import _kl_from_ref

    started_at = time.time()
    rng = np.random.RandomState(42)
    # 8 rows x 256 columns, Laplace-ish with a few outliers: the regime where
    # min/max collapses low-bit codes and the search earns its keep.
    weights = rng.laplace(0.0, 0.02, size=(8, 256)).astype(np.float32)
    weights[0, 5] = 0.9
    weights[3, 200] = -0.7
    weights[7, 64] = 0.0
    weights_mlx = mx.array(weights)
    cases = {}
    widths = [2, 3, 4, 8]
    for index, bits in enumerate(widths, 1):
        packed, scales, biases = quantize_best(weights_mlx, 64, bits)
        stock_packed, stock_scales, stock_biases = mx.quantize(
            weights_mlx, group_size=64, bits=bits, mode="affine")
        mx.eval(packed, scales, biases, stock_packed, stock_scales, stock_biases)
        cases[str(bits)] = {
            "search_packed_words": np.array(packed).astype(np.uint32).flatten().tolist(),
            "search_scales": np.array(scales.astype(mx.float32)).flatten().tolist(),
            "search_biases": np.array(biases.astype(mx.float32)).flatten().tolist(),
            "stock_packed_words": np.array(stock_packed).astype(np.uint32).flatten().tolist(),
            "stock_scales": np.array(stock_scales.astype(mx.float32)).flatten().tolist(),
            "stock_biases": np.array(stock_biases.astype(mx.float32)).flatten().tolist(),
        }
        _report("encoder", index, len(widths) + 1, started_at)

    reference_logits = rng.randn(1, 4, 16).astype(np.float32)
    probe_logits = (reference_logits + 0.1 * rng.randn(1, 4, 16)).astype(np.float32)
    kl = float(_kl_from_ref(mx.array(probe_logits), mx.array(reference_logits)).item())
    _report("encoder", len(widths) + 1, len(widths) + 1, started_at)
    _write(output_directory / "encoder_parity.json", {
        "optiq_version": _optiq_version(),
        "mlx_version": mx.__version__,
        "rows": 8,
        "columns": 256,
        "group_size": 64,
        "weights_row_major": weights.flatten().tolist(),
        "cases": cases,
        "kl_divergence": {
            "reference_logits": reference_logits.flatten().tolist(),
            "probe_logits": probe_logits.flatten().tolist(),
            "shape": [1, 4, 16],
            "kl_reference_vs_probe": kl,
        },
    })


def _optiq_version() -> str:
    import optiq

    return optiq.__version__


def main() -> int:
    output_directory = Path(__file__).resolve().parent
    started_at = time.time()
    stages = [
        generate_allocation_parity,
        generate_taxonomy_parity,
        generate_static_scorer_parity,
        generate_encoder_parity,
    ]
    for index, stage in enumerate(stages, 1):
        stage(output_directory)
        _report("fixtures", index, len(stages), started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
