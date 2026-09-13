#!/usr/bin/env python3
"""Export the expert-route predictor MLP as 1x1 convolutions for Core ML.

The Neural Engine wants BC1S activations and 1x1 conv instead of dense
linear layers. Training stays on CPU; this file only freezes a snapshot
for inference. Compute units are CPU_AND_NE and never ALL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def _conv_weight(row_major: list[float], output_channels: int, input_channels: int) -> np.ndarray:
    return np.asarray(row_major, dtype=np.float16).reshape(output_channels, input_channels, 1, 1)


def export_predictor(weights_document: dict, output_package: Path) -> None:
    layer_count = int(weights_document["layer_count"])
    expert_count = int(weights_document["expert_count"])
    input_dim = int(weights_document["input_dim"])
    hidden_dim = int(weights_document["hidden_dim"])
    layers = weights_document["layers"]
    if len(layers) != layer_count:
        raise ValueError("layer_count does not match layers")

    conv1_weights = [
        _conv_weight(layer["input_weights"], hidden_dim, input_dim) for layer in layers
    ]
    conv1_biases = [np.asarray(layer["input_bias"], dtype=np.float16) for layer in layers]
    conv2_weights = [
        _conv_weight(layer["output_weights"], expert_count, hidden_dim) for layer in layers
    ]
    conv2_biases = [np.asarray(layer["output_bias"], dtype=np.float16) for layer in layers]

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(layer_count, input_dim, 1, 1), dtype=types.fp16)
        ]
    )
    def predictor_program(head_inputs):
        layer_logits = []
        for layer_index in range(layer_count):
            layer_input = mb.slice_by_index(
                x=head_inputs,
                begin=[layer_index, 0, 0, 0],
                end=[layer_index + 1, input_dim, 1, 1],
                name=f"layer_{layer_index}_input",
            )
            hidden = mb.conv(
                x=layer_input,
                weight=conv1_weights[layer_index],
                bias=conv1_biases[layer_index],
                pad_type="valid",
                name=f"layer_{layer_index}_fc1",
            )
            hidden = mb.leaky_relu(
                x=hidden,
                alpha=np.float16(0.01),
                name=f"layer_{layer_index}_lrelu",
            )
            logits = mb.conv(
                x=hidden,
                weight=conv2_weights[layer_index],
                bias=conv2_biases[layer_index],
                pad_type="valid",
                name=f"layer_{layer_index}_fc2",
            )
            layer_logits.append(logits)
        return mb.concat(values=layer_logits, axis=0, name="logits")

    converted = ct.convert(
        predictor_program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    output_package.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output_package))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--output-mlpackage", type=Path, required=True)
    arguments = parser.parse_args()
    weights_document = json.loads(arguments.weights_json.read_text())
    export_predictor(weights_document, arguments.output_mlpackage)
    print(f"wrote {arguments.output_mlpackage}")


if __name__ == "__main__":
    main()
