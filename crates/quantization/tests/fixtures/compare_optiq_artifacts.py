#!/usr/bin/env python3
"""Compare two OptiQ-format artifact directories file by file and tensor by tensor.

Usage: compare_optiq_artifacts.py <reference_dir> <candidate_dir> [--ignore-json-key KEY ...]
Exit code 0 when every compared item is identical, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

JSON_DOCUMENTS = ("config.json", "optiq/metadata.json", "optiq/sensitivity.json",
                  "model.safetensors.index.json", "generation_config.json")


def json_differences(reference, candidate, path="") -> list[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        found = []
        for key in sorted(set(reference) | set(candidate)):
            where = f"{path}.{key}" if path else key
            if key not in candidate:
                found.append(f"missing in candidate: {where}")
            elif key not in reference:
                found.append(f"extra in candidate: {where}")
            else:
                found += json_differences(reference[key], candidate[key], where)
        return found
    if isinstance(reference, float) or isinstance(candidate, float):
        try:
            if abs(float(reference) - float(candidate)) <= 1e-9 * max(1.0, abs(float(reference))):
                return []
        except (TypeError, ValueError):
            pass
    return [] if reference == candidate else [f"{path}: {reference!r} != {candidate!r}"]


def load_tensors(directory: Path) -> dict:
    tensors = {}
    for shard in sorted(directory.glob("*.safetensors")):
        for name, tensor in mx.load(str(shard)).items():
            tensors[name] = (shard.name, tensor)
    return tensors


def comparable(tensor) -> np.ndarray:
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float32)
    return np.array(tensor)


def main() -> int:
    arguments = sys.argv[1:]
    ignored_keys = set()
    while "--ignore-json-key" in arguments:
        position = arguments.index("--ignore-json-key")
        ignored_keys.add(arguments[position + 1])
        del arguments[position:position + 2]
    reference_dir, candidate_dir = Path(arguments[0]), Path(arguments[1])
    problem_count = 0

    reference_files = {p.relative_to(reference_dir).as_posix() for p in reference_dir.rglob("*") if p.is_file()}
    candidate_files = {p.relative_to(candidate_dir).as_posix() for p in candidate_dir.rglob("*") if p.is_file()}
    for label, names in (("only in reference", reference_files - candidate_files),
                         ("only in candidate", candidate_files - reference_files)):
        if names:
            problem_count += len(names)
            print(f"FILES {label}: {sorted(names)}")

    for document in JSON_DOCUMENTS:
        reference_path, candidate_path = reference_dir / document, candidate_dir / document
        if not (reference_path.is_file() and candidate_path.is_file()):
            continue
        differences = [d for d in json_differences(json.loads(reference_path.read_text()),
                                                   json.loads(candidate_path.read_text()))
                       if not any(key in d for key in ignored_keys)]
        problem_count += len(differences)
        print(f"JSON {document}: {'identical' if not differences else f'{len(differences)} differences'}")
        for difference in differences[:12]:
            print(f"   {difference}")

    reference_tensors, candidate_tensors = load_tensors(reference_dir), load_tensors(candidate_dir)
    for label, names in (("only in reference", set(reference_tensors) - set(candidate_tensors)),
                         ("only in candidate", set(candidate_tensors) - set(reference_tensors))):
        if names:
            problem_count += len(names)
            print(f"TENSORS {label}: {len(names)} e.g. {sorted(names)[:6]}")
    shared = sorted(set(reference_tensors) & set(candidate_tensors))
    identical, mismatched = 0, []
    started_at, last_report = time.time(), time.time()
    for position, name in enumerate(shared, 1):
        (reference_shard, reference), (candidate_shard, candidate) = reference_tensors[name], candidate_tensors[name]
        if reference.dtype != candidate.dtype or reference.shape != candidate.shape:
            mismatched.append(f"{name}: {reference.dtype}{reference.shape} vs {candidate.dtype}{candidate.shape}")
        elif reference_shard.startswith("optiq_vision") != candidate_shard.startswith("optiq_vision"):
            mismatched.append(f"{name}: stored in {reference_shard} vs {candidate_shard}")
        else:
            reference_values, candidate_values = comparable(reference), comparable(candidate)
            if np.array_equal(reference_values, candidate_values):
                identical += 1
            else:
                delta = np.abs(reference_values.astype(np.float64) - candidate_values.astype(np.float64))
                share = float((reference_values != candidate_values).mean())
                mismatched.append(f"{name}: values differ (max abs {delta.max():.3e}, {100 * share:.3f}% of elements)")
        if time.time() - last_report >= 5.0:
            elapsed = time.time() - started_at
            print(f"  [compare] {position}/{len(shared)} ({100.0 * position / len(shared):.0f}%) "
                  f"elapsed {elapsed:.0f}s eta {elapsed / position * (len(shared) - position):.0f}s", flush=True)
            last_report = time.time()
    problem_count += len(mismatched)
    print(f"TENSORS shared {len(shared)}: {identical} bit-identical, {len(mismatched)} mismatched")
    for line in mismatched[:25]:
        print(f"   {line}")
    print("RESULT:", "IDENTICAL" if problem_count == 0 else f"{problem_count} differences")
    return 0 if problem_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
