//! Writes one prefill projection as a Core ML ML Program (iOS 18 opset), the
//! format that can keep weights compressed.
//!
//! Each weight is an fp16 `const`, a `constexpr_blockwise_shift_scale`, or a
//! `constexpr_lut_to_dense` over affine-quantized codes; the projection itself
//! is the same 1 × 1 convolution, with the same optional contraction split, as
//! the legacy fp16 document ([`crate::documents::neural_network`]).
//!
//! Measured on the MLP gate and up projection (100.7 million weights, issue
//! #161): the ML Program format alone cuts the process-footprint growth of a
//! Neural Engine load from 1,493 MiB (`NeuralNetwork`, fp16) to 392 MiB, and
//! 8-bit lookup tables shared by 16 rows cut it to 155 MiB at the same speed.
//! Core ML places shift-scale weights on the CPU in every block shape, and
//! lookup tables on the Neural Engine only when each table spans several
//! rows (one table per row runs about 14 times slower), so the 4-bit
//! checkpoint's own 64-input blocks cannot stay 4-bit on the Neural Engine.

use crate::documents::mil_operations::{
    SPECIFICATION_VERSION, activation_type, constant_operation, contraction_sum_operations,
    convolution_operation, convolution_parameter_constants, program, weight_blob_value,
};
use crate::documents::model_interface::{
    ACTIVATIONS_FEATURE_NAME, ModelDocumentError, PROJECTED_FEATURE_NAME, model_description,
};
use crate::documents::package_layout::MlProgramPackageContents;
use crate::program_shape::ProgramShape;
use crate::weights::affine_quantized::AffineQuantizedWeights;
use crate::weights::row_major_block::row_block;
use crate::wire::mil_program::{MilDataType, MilOperation, int32_tensor_value, tensor_value_type};
use crate::wire::protobuf::{append_bytes_field, append_varint_field};
use crate::wire::weight_blob::WeightBlobFile;

/// The weights an ML Program projection is built from.
#[derive(Clone, Copy, Debug)]
pub enum MlProgramWeights<'weights> {
    /// Row-major `[output_channels][input_channels]` fp16 bit patterns.
    Fp16(&'weights [u16]),
    /// Codes with one fp16 scale and offset per quantization block
    /// (`constexpr_blockwise_shift_scale`), kept compressed in the package.
    BlockwiseShiftScale(&'weights AffineQuantizedWeights),
    /// The same codes as indices into one fp16 lookup table per block
    /// (`constexpr_lut_to_dense`), which reproduces the checkpoint exactly;
    /// palettized weights are the form Apple documents for the Neural Engine.
    LookupTable(&'weights AffineQuantizedWeights),
}

pub fn encode_ml_program_projection(
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
) -> Result<MlProgramPackageContents, ModelDocumentError> {
    validate_weights(shape, weights, contraction_part_count)?;
    let mut weight_blob_file = WeightBlobFile::new();
    let operations = projection_operations(
        shape,
        weights,
        contraction_part_count,
        &mut weight_blob_file,
    );
    let mut model_document = Vec::new();
    append_varint_field(&mut model_document, 1, SPECIFICATION_VERSION);
    append_bytes_field(&mut model_document, 2, &model_description(shape));
    append_bytes_field(&mut model_document, 502, &program(shape, &operations));
    Ok(MlProgramPackageContents {
        model_document,
        weight_blob_file: weight_blob_file.into_bytes(),
    })
}

fn validate_weights(
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
) -> Result<(), ModelDocumentError> {
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
    match weights {
        MlProgramWeights::Fp16(fp16_weights) => validate_fp16_weights(shape, fp16_weights),
        MlProgramWeights::BlockwiseShiftScale(quantized_weights)
        | MlProgramWeights::LookupTable(quantized_weights) => {
            validate_quantized_weights(shape, quantized_weights, contraction_part_count)
        }
    }
}

fn validate_fp16_weights(
    shape: ProgramShape,
    fp16_weights: &[u16],
) -> Result<(), ModelDocumentError> {
    if fp16_weights.len() == shape.weight_element_count() {
        return Ok(());
    }
    Err(ModelDocumentError::WeightCountMismatch {
        expected: shape.weight_element_count(),
        actual: fp16_weights.len(),
    })
}

fn validate_quantized_weights(
    shape: ProgramShape,
    quantized_weights: &AffineQuantizedWeights,
    contraction_part_count: usize,
) -> Result<(), ModelDocumentError> {
    if quantized_weights.output_channels() != shape.output_channels()
        || quantized_weights.input_channels() != shape.input_channels()
    {
        return Err(ModelDocumentError::QuantizedShapeMismatch {
            output_channels: quantized_weights.output_channels(),
            input_channels: quantized_weights.input_channels(),
        });
    }
    let part_width = shape.input_channels() / contraction_part_count;
    let block_input_columns = quantized_weights.layout().block_input_columns;
    if !part_width.is_multiple_of(block_input_columns) {
        return Err(ModelDocumentError::ContractionPartCutsQuantizationBlock {
            part_width,
            block_input_columns,
        });
    }
    Ok(())
}

fn projection_operations(
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<Vec<u8>> {
    let mut operations = convolution_parameter_constants();
    let part_width = shape.input_channels() / contraction_part_count;
    if contraction_part_count > 1 {
        operations.extend(contraction_split_operations(shape, contraction_part_count));
    }
    let mut partial_projection_names = Vec::with_capacity(contraction_part_count);
    for (part_index, part_input_name) in part_input_names(contraction_part_count).iter().enumerate()
    {
        let weight_name = format!("weight_part_{part_index}");
        operations.push(part_weight_operation(
            &weight_name,
            shape,
            weights,
            part_index * part_width,
            part_width,
            weight_blob_file,
        ));
        let partial_projection_name = partial_projection_name(part_index, contraction_part_count);
        operations.push(convolution_operation(
            shape,
            part_input_name,
            &weight_name,
            &partial_projection_name,
        ));
        partial_projection_names.push(partial_projection_name);
    }
    operations.extend(contraction_sum_operations(
        shape,
        &partial_projection_names,
        "contraction_sum",
        PROJECTED_FEATURE_NAME,
    ));
    operations
}

fn activations_part_name(part_index: usize) -> String {
    format!("activations_part_{part_index}")
}

/// What each part's convolution reads: the whole input when the contraction
/// is not split, otherwise the matching output of the split.
fn part_input_names(contraction_part_count: usize) -> Vec<String> {
    if contraction_part_count == 1 {
        return vec![ACTIVATIONS_FEATURE_NAME.to_owned()];
    }
    (0..contraction_part_count)
        .map(activations_part_name)
        .collect()
}

/// A lone convolution writes the model output directly; split parts write
/// partial sums for the add chain.
fn partial_projection_name(part_index: usize, contraction_part_count: usize) -> String {
    if contraction_part_count == 1 {
        PROJECTED_FEATURE_NAME.to_owned()
    } else {
        format!("partial_projection_{part_index}")
    }
}

fn contraction_split_operations(
    shape: ProgramShape,
    contraction_part_count: usize,
) -> Vec<Vec<u8>> {
    let part_width = shape.input_channels() / contraction_part_count;
    let part_type = activation_type(part_width, shape.token_count());
    let mut split_operation = MilOperation::new("split", "contraction_split")
        .input_named("x", ACTIVATIONS_FEATURE_NAME)
        .input_named("num_splits", "split_count")
        .input_named("axis", "split_axis");
    for part_index in 0..contraction_part_count {
        split_operation = split_operation.output(&activations_part_name(part_index), &part_type);
    }
    vec![
        constant_operation(
            "split_count",
            &int32_tensor_value(&[contraction_part_count as i32], &[]),
            MilDataType::Int32,
            &[],
        ),
        constant_operation(
            "split_axis",
            &int32_tensor_value(&[1], &[]),
            MilDataType::Int32,
            &[],
        ),
        split_operation.into_bytes(),
    ]
}

fn part_weight_operation(
    weight_name: &str,
    shape: ProgramShape,
    weights: MlProgramWeights<'_>,
    first_input_channel: usize,
    part_width: usize,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<u8> {
    match weights {
        MlProgramWeights::Fp16(fp16_weights) => {
            let part_weights = row_block(
                fp16_weights,
                shape.input_channels(),
                first_input_channel,
                part_width,
            );
            let weight_dimensions = [shape.output_channels(), part_width, 1, 1];
            let blob_offset = weight_blob_file.append_fp16(&part_weights);
            constant_operation(
                weight_name,
                &weight_blob_value(MilDataType::Float16, &weight_dimensions, blob_offset),
                MilDataType::Float16,
                &weight_dimensions,
            )
        }
        MlProgramWeights::BlockwiseShiftScale(quantized_weights) => {
            let part_weights = quantized_weights.contraction_part(first_input_channel, part_width);
            blockwise_shift_scale_operation(weight_name, &part_weights, weight_blob_file)
        }
        MlProgramWeights::LookupTable(quantized_weights) => {
            let part_weights = quantized_weights.contraction_part(first_input_channel, part_width);
            lookup_table_operation(weight_name, &part_weights, weight_blob_file)
        }
    }
}

fn weight_dimensions_of(part_weights: &AffineQuantizedWeights) -> [usize; 4] {
    [
        part_weights.output_channels(),
        part_weights.input_channels(),
        1,
        1,
    ]
}

/// One entry per quantization block, laid out like the weight so each
/// dimension divides the weight's: `[O / block rows, I / block columns, 1, 1]`.
fn block_grid_dimensions_of(part_weights: &AffineQuantizedWeights) -> [usize; 4] {
    let [block_rows, blocks_per_row] = part_weights.block_grid();
    [block_rows, blocks_per_row, 1, 1]
}

/// Appends the codes at their own width and returns their MIL type and blob
/// offset.
fn append_codes(
    part_weights: &AffineQuantizedWeights,
    weight_blob_file: &mut WeightBlobFile,
) -> (MilDataType, u64) {
    if part_weights.layout().code_bits == 8 {
        (
            MilDataType::UnsignedInt8,
            weight_blob_file.append_uint8(part_weights.codes()),
        )
    } else {
        (
            MilDataType::UnsignedInt4,
            weight_blob_file.append_uint4(part_weights.codes()),
        )
    }
}

/// `constexpr_lut_to_dense`: indices `[O, I, 1, 1]` and a lookup table of
/// rank two higher, `[O / block rows, I / block columns, 1, 1, entries, 1]`,
/// whose leading dimensions divide the indices' so each table covers one
/// quantization block.
fn lookup_table_operation(
    weight_name: &str,
    part_weights: &AffineQuantizedWeights,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<u8> {
    let weight_dimensions = weight_dimensions_of(part_weights);
    let [block_rows, blocks_per_row, ..] = block_grid_dimensions_of(part_weights);
    let lookup_table_dimensions = [
        block_rows,
        blocks_per_row,
        1,
        1,
        part_weights.layout().code_count(),
        1,
    ];
    let (indices_type, indices_offset) = append_codes(part_weights, weight_blob_file);
    let lookup_table_offset =
        weight_blob_file.append_fp16(&part_weights.block_lookup_tables_fp16());
    MilOperation::new("constexpr_lut_to_dense", weight_name)
        .input_value(
            "indices",
            &weight_blob_value(indices_type, &weight_dimensions, indices_offset),
        )
        .input_value(
            "lut",
            &weight_blob_value(
                MilDataType::Float16,
                &lookup_table_dimensions,
                lookup_table_offset,
            ),
        )
        .output(
            weight_name,
            &tensor_value_type(MilDataType::Float16, &weight_dimensions),
        )
        .into_bytes()
}

fn blockwise_shift_scale_operation(
    weight_name: &str,
    part_weights: &AffineQuantizedWeights,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<u8> {
    let weight_dimensions = weight_dimensions_of(part_weights);
    let block_grid_dimensions = block_grid_dimensions_of(part_weights);
    let (codes_type, codes_offset) = append_codes(part_weights, weight_blob_file);
    let scales_offset = weight_blob_file.append_fp16(part_weights.fp16_scales());
    let offsets_offset = weight_blob_file.append_fp16(&part_weights.shift_scale_offsets_fp16());
    MilOperation::new("constexpr_blockwise_shift_scale", weight_name)
        .input_value(
            "data",
            &weight_blob_value(codes_type, &weight_dimensions, codes_offset),
        )
        .input_value(
            "scale",
            &weight_blob_value(MilDataType::Float16, &block_grid_dimensions, scales_offset),
        )
        .input_value(
            "offset",
            &weight_blob_value(MilDataType::Float16, &block_grid_dimensions, offsets_offset),
        )
        .output(
            weight_name,
            &tensor_value_type(MilDataType::Float16, &weight_dimensions),
        )
        .into_bytes()
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
