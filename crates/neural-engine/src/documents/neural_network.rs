//! Writes one prefill projection as a Core ML `NeuralNetwork` model document.
//!
//! The projection is expressed as a 1 × 1 convolution over a
//! `[1, input_channels, 1, tokens]` tensor because the Neural Engine runs
//! convolutions at roughly 1.8 times its general matrix-multiply rate on
//! M5-class silicon (18.8 versus 10.2 TFLOP/s measured in #540); the
//! arithmetic is identical to `weights · activations`.
//!
//! Weights are stored as fp16, the Neural Engine's native compute type, so the
//! document adds no quantization of its own. Inputs and outputs are declared
//! fp16 so IOSurface-backed half-precision buffers can be handed to Core ML
//! without a conversion inside every prediction.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::documents::model_interface::{
    ACTIVATIONS_FEATURE_NAME, ModelDocumentError, PROJECTED_FEATURE_NAME, model_description,
};
use crate::program_shape::ProgramShape;
use crate::weights::row_major_block::row_block;
use crate::wire::protobuf::{append_bytes_field, append_string_field, append_varint_field};

// Core ML 6 (specification version 7) is the first to accept fp16 multi-array
// inputs and outputs.
const SPECIFICATION_VERSION: u64 = 7;
// Rank-4 multi-arrays map to [batch, channels, height, width].
const EXACT_ARRAY_MAPPING: u64 = 1;

/// Encodes `fp16_weights` (row-major `[output_channels][input_channels]`
/// binary16 bit patterns) as a complete `.mlmodel` document.
///
/// With `contraction_part_count` above one, the input channels are split into
/// that many equal parts, each part is projected by its own convolution over
/// the matching weight columns, and the partial results are added. The sum is
/// the same projection; it exists because the Neural Engine runs a
/// 2,048-deep contraction at 18.6 TFLOP/s but a 4,096-deep one at 9.7
/// (shape sweep, issue #161).
pub fn encode_projection_document(
    shape: ProgramShape,
    fp16_weights: &[u16],
    contraction_part_count: usize,
) -> Result<Vec<u8>, ModelDocumentError> {
    if fp16_weights.len() != shape.weight_element_count() {
        return Err(ModelDocumentError::WeightCountMismatch {
            expected: shape.weight_element_count(),
            actual: fp16_weights.len(),
        });
    }
    if contraction_part_count == 0
        || !shape
            .input_channels()
            .is_multiple_of(contraction_part_count)
    {
        return Err(ModelDocumentError::UnevenContractionSplit {
            input_channels: shape.input_channels(),
            contraction_part_count,
        });
    }
    let mut network = Vec::new();
    if contraction_part_count == 1 {
        let whole_projection = ConvolutionSpecification {
            layer_name: "projection",
            input_name: ACTIVATIONS_FEATURE_NAME,
            output_name: PROJECTED_FEATURE_NAME.to_owned(),
            output_channels: shape.output_channels(),
            kernel_channels: shape.input_channels(),
        };
        append_bytes_field(
            &mut network,
            1,
            &convolution_layer(&whole_projection, fp16_weights),
        );
    } else {
        append_contraction_split_layers(&mut network, shape, fp16_weights, contraction_part_count);
    }
    append_varint_field(&mut network, 5, EXACT_ARRAY_MAPPING);

    let mut model_document = Vec::new();
    append_varint_field(&mut model_document, 1, SPECIFICATION_VERSION);
    append_bytes_field(&mut model_document, 2, &model_description(shape));
    append_bytes_field(&mut model_document, 500, &network);
    Ok(model_document)
}

struct ConvolutionSpecification<'names> {
    layer_name: &'names str,
    input_name: &'names str,
    output_name: String,
    output_channels: usize,
    kernel_channels: usize,
}

fn convolution_layer(
    specification: &ConvolutionSpecification<'_>,
    fp16_weights: &[u16],
) -> Vec<u8> {
    let mut convolution = Vec::new();
    append_varint_field(&mut convolution, 1, specification.output_channels as u64);
    append_varint_field(&mut convolution, 2, specification.kernel_channels as u64);
    append_varint_field(&mut convolution, 10, 1);
    for (field_number, unit_extent) in [(20, 1), (20, 1), (30, 1), (30, 1)] {
        append_varint_field(&mut convolution, field_number, unit_extent);
    }
    // Field 50 selects "valid" padding, which for a 1 × 1 kernel means none.
    append_bytes_field(&mut convolution, 50, &[]);
    append_bytes_field(&mut convolution, 90, &fp16_weight_parameters(fp16_weights));

    let mut layer = Vec::new();
    append_string_field(&mut layer, 1, specification.layer_name);
    append_string_field(&mut layer, 2, specification.input_name);
    append_string_field(&mut layer, 3, &specification.output_name);
    append_bytes_field(&mut layer, 100, &convolution);
    layer
}

/// Split (field 330) → one convolution per part → add (field 230).
fn append_contraction_split_layers(
    network: &mut Vec<u8>,
    shape: ProgramShape,
    fp16_weights: &[u16],
    contraction_part_count: usize,
) {
    let part_width = shape.input_channels() / contraction_part_count;
    let mut split_layer = Vec::new();
    append_string_field(&mut split_layer, 1, "contraction_split");
    append_string_field(&mut split_layer, 2, ACTIVATIONS_FEATURE_NAME);
    for part_index in 0..contraction_part_count {
        append_string_field(
            &mut split_layer,
            3,
            &format!("activations_part_{part_index}"),
        );
    }
    let mut split_parameters = Vec::new();
    append_varint_field(&mut split_parameters, 1, contraction_part_count as u64);
    append_bytes_field(&mut split_layer, 330, &split_parameters);
    append_bytes_field(network, 1, &split_layer);

    let mut add_layer = Vec::new();
    append_string_field(&mut add_layer, 1, "contraction_sum");
    for part_index in 0..contraction_part_count {
        let input_name = format!("activations_part_{part_index}");
        let part_projection = ConvolutionSpecification {
            layer_name: &format!("projection_part_{part_index}"),
            input_name: &input_name,
            output_name: format!("partial_projection_{part_index}"),
            output_channels: shape.output_channels(),
            kernel_channels: part_width,
        };
        let part_weights = row_block(
            fp16_weights,
            shape.input_channels(),
            part_index * part_width,
            part_width,
        );
        append_bytes_field(
            network,
            1,
            &convolution_layer(&part_projection, &part_weights),
        );
        append_string_field(&mut add_layer, 2, &part_projection.output_name);
    }
    append_string_field(&mut add_layer, 3, PROJECTED_FEATURE_NAME);
    append_bytes_field(&mut add_layer, 230, &[]);
    append_bytes_field(network, 1, &add_layer);
}

fn fp16_weight_parameters(fp16_weights: &[u16]) -> Vec<u8> {
    let mut little_endian_halves = Vec::with_capacity(fp16_weights.len() * 2);
    for half_bits in fp16_weights {
        little_endian_halves.extend_from_slice(&half_bits.to_le_bytes());
    }
    let mut weight_parameters = Vec::new();
    append_bytes_field(&mut weight_parameters, 2, &little_endian_halves);
    weight_parameters
}

/// Writes `projection.mlmodel` inside `model_directory` and returns its path.
pub fn write_model_document(model_directory: &Path, model_document: &[u8]) -> io::Result<PathBuf> {
    let model_path = model_directory.join("projection.mlmodel");
    fs::write(&model_path, model_document)?;
    Ok(model_path)
}

#[cfg(test)]
#[path = "neural_network_tests.rs"]
mod tests;
