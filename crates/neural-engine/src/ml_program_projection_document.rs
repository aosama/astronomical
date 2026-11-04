//! Writes one prefill projection as a Core ML ML Program (iOS 18 opset), the
//! format that can keep weights compressed.
//!
//! Each weight is an fp16 `const`, a `constexpr_blockwise_shift_scale`, or a
//! `constexpr_lut_to_dense` over affine-quantized codes; the projection itself
//! is the same 1 × 1 convolution, with the same optional contraction split, as
//! the legacy fp16 document ([`crate::core_ml_projection_document`]).
//!
//! Measured on the MLP gate and up projection (100.7 million weights, issue
//! #161): the ML Program format alone cuts the process-footprint growth of a
//! Neural Engine load from 1,493 MiB (`NeuralNetwork`, fp16) to 392 MiB, and
//! 8-bit lookup tables shared by 16 rows cut it to 155 MiB at the same speed.
//! Core ML places shift-scale weights on the CPU in every block shape, and
//! lookup tables on the Neural Engine only when each table spans several
//! rows (one table per row runs about 14 times slower), so the 4-bit
//! checkpoint's own 64-input blocks cannot stay 4-bit on the Neural Engine.

use crate::affine_quantized_weights::AffineQuantizedWeights;
use crate::mil_program_wire::{
    MilDataType, MilOperation, append_map_entry, blob_file_tensor_value, int32_tensor_value,
    named_value_type, string_scalar_value, tensor_value_type,
};
use crate::ml_program_weight_blob::WeightBlobFile;
use crate::projection_model_interface::{
    ACTIVATIONS_FEATURE_NAME, PROJECTED_FEATURE_NAME, ProjectionDocumentError, model_description,
};
use crate::projection_shape::ProjectionShape;
use crate::protobuf_wire::{append_bytes_field, append_string_field, append_varint_field};
use crate::row_major_block::row_block;

// Specification version 9 is iOS 18 / macOS 15, the first with block-wise
// 4-bit weights; its MIL opset is "CoreML8".
pub(crate) const SPECIFICATION_VERSION: u64 = 9;
const OPSET_NAME: &str = "CoreML8";
const MAIN_FUNCTION_NAME: &str = "main";
/// Where the program expects its weights, relative to the model package.
pub const WEIGHT_BLOB_FILE_NAME: &str = "@model_path/weights/weight.bin";

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

/// The two files of an `.mlpackage`: the model document and its weights.
pub struct MlProgramPackageContents {
    pub model_document: Vec<u8>,
    pub weight_blob_file: Vec<u8>,
}

pub fn encode_ml_program_projection(
    shape: ProjectionShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
) -> Result<MlProgramPackageContents, ProjectionDocumentError> {
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
    shape: ProjectionShape,
    weights: MlProgramWeights<'_>,
    contraction_part_count: usize,
) -> Result<(), ProjectionDocumentError> {
    if contraction_part_count == 0
        || !shape
            .input_channels()
            .is_multiple_of(contraction_part_count)
    {
        return Err(ProjectionDocumentError::UnevenContractionSplit {
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
    shape: ProjectionShape,
    fp16_weights: &[u16],
) -> Result<(), ProjectionDocumentError> {
    if fp16_weights.len() == shape.weight_element_count() {
        return Ok(());
    }
    Err(ProjectionDocumentError::WeightCountMismatch {
        expected: shape.weight_element_count(),
        actual: fp16_weights.len(),
    })
}

fn validate_quantized_weights(
    shape: ProjectionShape,
    quantized_weights: &AffineQuantizedWeights,
    contraction_part_count: usize,
) -> Result<(), ProjectionDocumentError> {
    if quantized_weights.output_channels() != shape.output_channels()
        || quantized_weights.input_channels() != shape.input_channels()
    {
        return Err(ProjectionDocumentError::QuantizedShapeMismatch {
            output_channels: quantized_weights.output_channels(),
            input_channels: quantized_weights.input_channels(),
        });
    }
    let part_width = shape.input_channels() / contraction_part_count;
    let block_input_columns = quantized_weights.layout().block_input_columns;
    if !part_width.is_multiple_of(block_input_columns) {
        return Err(
            ProjectionDocumentError::ContractionPartCutsQuantizationBlock {
                part_width,
                block_input_columns,
            },
        );
    }
    Ok(())
}

fn projection_operations(
    shape: ProjectionShape,
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

// A 1 × 1 kernel with unit stride and no padding is exactly a matrix
// multiply over channels; these are the values Apple's converter writes as
// conv defaults, emitted once and shared by every part.
const CONVOLUTION_PARAMETER_CONSTANTS: [&str; 5] = [
    "conv_strides",
    "conv_pad_type",
    "conv_pad",
    "conv_dilations",
    "conv_groups",
];

pub(crate) fn convolution_parameter_constants() -> Vec<Vec<u8>> {
    vec![
        constant_operation(
            "conv_strides",
            &int32_tensor_value(&[1, 1], &[2]),
            MilDataType::Int32,
            &[2],
        ),
        constant_operation(
            "conv_pad_type",
            &string_scalar_value("valid"),
            MilDataType::String,
            &[],
        ),
        constant_operation(
            "conv_pad",
            &int32_tensor_value(&[0, 0, 0, 0], &[4]),
            MilDataType::Int32,
            &[4],
        ),
        constant_operation(
            "conv_dilations",
            &int32_tensor_value(&[1, 1], &[2]),
            MilDataType::Int32,
            &[2],
        ),
        constant_operation(
            "conv_groups",
            &int32_tensor_value(&[1], &[]),
            MilDataType::Int32,
            &[],
        ),
    ]
}

pub(crate) fn constant_operation(
    constant_name: &str,
    encoded_value: &[u8],
    data_type: MilDataType,
    dimensions: &[usize],
) -> Vec<u8> {
    MilOperation::new("const", constant_name)
        .attribute("val", encoded_value)
        .output(constant_name, &tensor_value_type(data_type, dimensions))
        .into_bytes()
}

fn contraction_split_operations(
    shape: ProjectionShape,
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
    shape: ProjectionShape,
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

pub(crate) fn weight_blob_value(
    data_type: MilDataType,
    dimensions: &[usize],
    blob_offset: u64,
) -> Vec<u8> {
    blob_file_tensor_value(data_type, dimensions, WEIGHT_BLOB_FILE_NAME, blob_offset)
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

pub(crate) fn convolution_operation(
    shape: ProjectionShape,
    input_name: &str,
    weight_name: &str,
    output_name: &str,
) -> Vec<u8> {
    let mut convolution = MilOperation::new("conv", output_name)
        .input_named("x", input_name)
        .input_named("weight", weight_name);
    for (parameter_name, constant_name) in ["strides", "pad_type", "pad", "dilations", "groups"]
        .iter()
        .zip(CONVOLUTION_PARAMETER_CONSTANTS)
    {
        convolution = convolution.input_named(parameter_name, constant_name);
    }
    convolution
        .output(
            output_name,
            &activation_type(shape.output_channels(), shape.token_count()),
        )
        .into_bytes()
}

/// `partial_0 + partial_1 + …` as a chain of binary adds ending in
/// `final_output_name`; MIL's `add` takes exactly two operands.
pub(crate) fn contraction_sum_operations(
    shape: ProjectionShape,
    partial_projection_names: &[String],
    sum_name_prefix: &str,
    final_output_name: &str,
) -> Vec<Vec<u8>> {
    let projected_type = activation_type(shape.output_channels(), shape.token_count());
    let mut operations = Vec::new();
    let mut running_sum_name = partial_projection_names[0].clone();
    for (part_index, partial_projection_name) in partial_projection_names.iter().enumerate().skip(1)
    {
        let sum_name = if part_index + 1 == partial_projection_names.len() {
            final_output_name.to_owned()
        } else {
            format!("{sum_name_prefix}_{part_index}")
        };
        operations.push(
            MilOperation::new("add", &sum_name)
                .input_named("x", &running_sum_name)
                .input_named("y", partial_projection_name)
                .output(&sum_name, &projected_type)
                .into_bytes(),
        );
        running_sum_name = sum_name;
    }
    operations
}

pub(crate) fn activation_type(channel_count: usize, token_count: usize) -> Vec<u8> {
    tensor_value_type(MilDataType::Float16, &[1, channel_count, 1, token_count])
}

pub(crate) fn program(shape: ProjectionShape, operations: &[Vec<u8>]) -> Vec<u8> {
    let mut block = Vec::new();
    append_string_field(&mut block, 2, PROJECTED_FEATURE_NAME);
    for operation in operations {
        append_bytes_field(&mut block, 3, operation);
    }
    let mut function = Vec::new();
    append_bytes_field(
        &mut function,
        1,
        &named_value_type(
            ACTIVATIONS_FEATURE_NAME,
            &activation_type(shape.input_channels(), shape.token_count()),
        ),
    );
    append_string_field(&mut function, 2, OPSET_NAME);
    append_map_entry(&mut function, 3, OPSET_NAME, &block);
    let mut program = Vec::new();
    append_varint_field(&mut program, 1, 1);
    append_map_entry(&mut program, 2, MAIN_FUNCTION_NAME, &function);
    program
}

#[cfg(test)]
#[path = "ml_program_projection_document_tests.rs"]
mod tests;
