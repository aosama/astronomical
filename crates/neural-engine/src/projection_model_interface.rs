//! What every projection model document promises the native bridge, whatever
//! its format: one fp16 `activations` input, one fp16 `projected` output, both
//! `[1, channels, 1, tokens]`, and the errors raised while writing one.
//!
//! The legacy `NeuralNetwork` document and the ML Program both declare this
//! same interface, which is why one bridge loads either.

use thiserror::Error;

use crate::projection_shape::ProjectionShape;
use crate::protobuf_wire::{append_bytes_field, append_string_field, append_varint_field};

/// The input feature name the Objective-C bridge binds to.
pub const ACTIVATIONS_FEATURE_NAME: &str = "activations";
/// The output feature name the Objective-C bridge binds to.
pub const PROJECTED_FEATURE_NAME: &str = "projected";

const FLOAT16_ARRAY_DATA_TYPE: u64 = 0x10000 | 16;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProjectionDocumentError {
    #[error("the projection needs {expected} fp16 weights, got {actual}")]
    WeightCountMismatch { expected: usize, actual: usize },
    #[error(
        "{input_channels} input channels cannot be split into {contraction_part_count} equal parts"
    )]
    UnevenContractionSplit {
        input_channels: usize,
        contraction_part_count: usize,
    },
    #[error(
        "{input_channels} input channels cannot be cut into parts {contraction_part_width} wide"
    )]
    UnevenContractionPartWidth {
        input_channels: usize,
        contraction_part_width: usize,
    },
    #[error(
        "quantized weights are {output_channels} × {input_channels}, \
         which does not match the projection shape"
    )]
    QuantizedShapeMismatch {
        output_channels: usize,
        input_channels: usize,
    },
    #[error(
        "a {part_width}-channel contraction part would cut a {block_input_columns}-column quantization block"
    )]
    ContractionPartCutsQuantizationBlock {
        part_width: usize,
        block_input_columns: usize,
    },
    #[error("{output_channels} output rows cannot share tables of {rows_per_table} rows")]
    UnevenLookupTableRows {
        output_channels: usize,
        rows_per_table: usize,
    },
    #[error("{channel_count} channels cannot be rotated in blocks of {block_size}")]
    UnevenRotationBlocks {
        channel_count: usize,
        block_size: usize,
    },
}

/// The `ModelDescription` declaring the fp16 `activations` input and
/// `projected` output.
#[must_use]
pub fn model_description(shape: ProjectionShape) -> Vec<u8> {
    let mut description = Vec::new();
    append_bytes_field(
        &mut description,
        1,
        &fp16_feature(
            ACTIVATIONS_FEATURE_NAME,
            shape.input_channels(),
            shape.token_count(),
        ),
    );
    append_bytes_field(
        &mut description,
        10,
        &fp16_feature(
            PROJECTED_FEATURE_NAME,
            shape.output_channels(),
            shape.token_count(),
        ),
    );
    description
}

fn fp16_feature(feature_name: &str, channel_count: usize, token_count: usize) -> Vec<u8> {
    let mut array_type = Vec::new();
    for dimension in [1, channel_count, 1, token_count] {
        append_varint_field(&mut array_type, 1, dimension as u64);
    }
    append_varint_field(&mut array_type, 2, FLOAT16_ARRAY_DATA_TYPE);
    let mut feature_type = Vec::new();
    append_bytes_field(&mut feature_type, 5, &array_type);
    let mut feature = Vec::new();
    append_string_field(&mut feature, 1, feature_name);
    append_bytes_field(&mut feature, 3, &feature_type);
    feature
}

#[cfg(test)]
#[path = "projection_model_interface_tests.rs"]
mod tests;
