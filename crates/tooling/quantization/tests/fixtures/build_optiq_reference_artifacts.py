#!/usr/bin/env python3
"""Build the OptiQ reference artifacts the acceptance parity journeys compare against.

Run with a Python environment that has ``mlx-optiq`` installed (the journeys
were pinned against 0.8.2 with mlx-lm 0.32.2). Writes, under ``reference_root``:

``static_mixed/``                      the ``static`` method's artifact
``optiq_mixed/``                       the measured (KL sweep) artifact
``sensitivity_checkpoint.json``        the sweep OptiQ measured
``low_bit/static_mixed/``              the ``static`` method on a 2/3/4/8 ladder
``parity/calibration_sequences.json``  the token sequences the sweep forwarded
``parity/kl_evaluation.json``          OptiQ's KL evaluation of ``optiq_mixed/``
``parity/versions.json``               the package versions that produced them

mlx-optiq 0.8.2 calls mlx-lm's ``quantize_model(model, config, group_size,
bits, mode, quant_predicate)`` positionally. mlx-lm 0.32 inserted
``global_scale`` before ``quant_predicate``, so the predicate lands in
``global_scale``, no allocation is applied, and the artifact comes out
uniform while its metadata describes a mixed allocation. A reference built
that way is not the artifact OptiQ intends, so this script re-routes a
callable passed in that slot to ``quant_predicate`` and changes nothing else.

Usage: build_optiq_reference_artifacts.py <source_dir> <reference_root>
           [--target-bpw 5.0] [--candidate-bits 4,8] [--calibration-sequences 4]
           [--reference bf16] [--methods static,optiq] [--skip-low-bit]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx_lm.convert  # noqa: F401  (makes the submodule importable below)

CALIBRATION_SEQUENCE_LENGTH_TOKENS = 512

# The ladder and target that put modules at 2 and 3 bits, where OptiQ picks
# each group's range by search instead of from its minimum and maximum.
LOW_BIT_CANDIDATE_BITS = [2, 3, 4, 8]
LOW_BIT_TARGET_BPW = 3.5

convert_module = sys.modules["mlx_lm.convert"]
original_quantize_model = convert_module.quantize_model


def quantize_model_with_predicate(model, config, group_size=None, bits=None, mode="affine",
                                  *shifted, **keywords):
    if shifted and callable(shifted[0]):
        keywords["quant_predicate"] = shifted[0]
        shifted = shifted[1:]
    return original_quantize_model(model, config, group_size, bits, mode, *shifted, **keywords)


convert_module.quantize_model = quantize_model_with_predicate


def report(stage: str, done: int, total: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    remaining = elapsed / done * (total - done) if done else float("nan")
    print(f"[reference] {stage} {done}/{total} ({100.0 * done / total:.0f}%) "
          f"elapsed {elapsed:.0f}s eta {remaining:.0f}s", flush=True)


def write_calibration_sequences(source_dir: str, reference_root: Path, sequence_count: int) -> None:
    """The exact sequences ``run_llm_pipeline`` forwards: it draws
    ``max(2 * n, 16)`` and the sweep keeps the first ``n``."""
    from mlx_lm.utils import load_tokenizer
    from optiq.calibration.datasets import load_llm_calibration

    tokenizer = load_tokenizer(Path(source_dir), tokenizer_config_extra={"trust_remote_code": True})
    calibration_fn = load_llm_calibration(
        tokenizer, n_samples=max(sequence_count * 2, 16),
        seq_len=CALIBRATION_SEQUENCE_LENGTH_TOKENS, mix="optiq")
    sequences = [arguments[0].tolist()[0] for arguments, _ in calibration_fn()[:sequence_count]]
    parity_dir = reference_root / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    (parity_dir / "calibration_sequences.json").write_text(json.dumps({
        "sequence_length_tokens": CALIBRATION_SEQUENCE_LENGTH_TOKENS,
        "sequences": sequences,
    }))
    (parity_dir / "versions.json").write_text(json.dumps({
        "mlx-optiq": version("mlx-optiq"), "mlx-lm": version("mlx-lm"), "mlx": version("mlx"),
    }, indent=2))


def write_kl_evaluation(source_dir: str, reference_root: Path) -> None:
    """OptiQ's own evaluation of its measured artifact against the source,
    with its defaults (64 prompts of at most 256 tokens)."""
    from optiq.eval.kl import evaluate_kl

    evaluation = evaluate_kl(str(reference_root / "optiq_mixed"), reference_path=source_dir,
                             reference_mode="bf16")
    (reference_root / "parity" / "kl_evaluation.json").write_text(json.dumps({
        "mean_kl_divergence": evaluation.mean_kl,
        "median_kl_divergence": evaluation.median_kl,
        "ninety_fifth_percentile_kl_divergence": evaluation.p95_kl,
        "prompt_count": evaluation.n_prompts,
        "sequence_length_tokens": evaluation.seq_len,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source_dir")
    parser.add_argument("reference_root")
    parser.add_argument("--target-bpw", type=float, default=5.0)
    parser.add_argument("--candidate-bits", default="4,8")
    parser.add_argument("--calibration-sequences", type=int, default=4)
    parser.add_argument("--reference", default="bf16")
    parser.add_argument("--methods", default="static,optiq")
    parser.add_argument("--skip-low-bit", action="store_true")
    arguments = parser.parse_args()

    from optiq.models.llm import run_llm_pipeline

    reference_root = Path(arguments.reference_root)
    reference_root.mkdir(parents=True, exist_ok=True)
    methods = arguments.methods.split(",")
    started_at = time.time()
    write_calibration_sequences(arguments.source_dir, reference_root, arguments.calibration_sequences)
    for position, method in enumerate(methods, 1):
        run_llm_pipeline(
            model_name=arguments.source_dir,
            output_dir=str(reference_root),
            target_bpw=arguments.target_bpw,
            n_calibration=arguments.calibration_sequences,
            candidate_bits=[int(bits) for bits in arguments.candidate_bits.split(",")],
            skip_baselines=True,
            reference=arguments.reference,
            method=method,
        )
        report("methods", position, len(methods), started_at)
    if "optiq" in methods:
        write_kl_evaluation(arguments.source_dir, reference_root)
        report("KL evaluation", 1, 1, started_at)
    if not arguments.skip_low_bit:
        run_llm_pipeline(
            model_name=arguments.source_dir,
            output_dir=str(reference_root / "low_bit"),
            target_bpw=LOW_BIT_TARGET_BPW,
            candidate_bits=LOW_BIT_CANDIDATE_BITS,
            skip_baselines=True,
            method="static",
        )
        report("low-bit ladder", 1, 1, started_at)


if __name__ == "__main__":
    main()
