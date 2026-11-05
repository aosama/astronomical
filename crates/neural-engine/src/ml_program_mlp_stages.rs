//! One projection of an MLP chain, cut into contraction parts, and the `silu`
//! gating between projections. Both whole-MLP documents are built from these.

use crate::mil_operations::{
    activation_type, constant_operation, contraction_sum_operations, convolution_operation,
    split_axis_constant, split_count_constant, weight_blob_value,
};
use crate::mil_program_wire::{MilDataType, MilOperation, tensor_value_type};
use crate::ml_program_weight_blob::WeightBlobFile;
use crate::projection_model_interface::ProjectionDocumentError;
use crate::projection_shape::ProjectionShape;
use crate::row_major_block::row_block;
use crate::shared_lookup_tables::{LOOKUP_TABLE_ENTRY_COUNT, SharedLookupTables};

pub(crate) const GATE_AND_UP_NAME: &str = "gate_and_up";
const GATE_NAME: &str = "gate";
const UP_NAME: &str = "up";
const SIGMOID_NAME: &str = "gate_sigmoid";
const SILU_NAME: &str = "gate_silu";
pub(crate) const GATED_NAME: &str = "gated";

/// How a stage stores each part's weights in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageWeightEncoding {
    Fp16,
    /// 8-bit codes into tables that each span this many output rows.
    SharedLookupTables {
        rows_per_table: usize,
    },
}

/// One projection of the chain, cut into contraction parts.
pub(crate) struct ProjectionStage<'stage> {
    pub(crate) stage_name: &'stage str,
    pub(crate) input_name: &'stage str,
    pub(crate) output_name: &'stage str,
    pub(crate) shape: ProjectionShape,
    pub(crate) fp16_weights: &'stage [u16],
    pub(crate) part_count: usize,
    pub(crate) weight_encoding: StageWeightEncoding,
}

pub(crate) fn validated_stage_shape(
    input_channels: usize,
    output_channels: usize,
    token_count: usize,
    fp16_weights: &[u16],
    contraction_part_width: usize,
) -> Result<ProjectionShape, ProjectionDocumentError> {
    if contraction_part_width == 0 || !input_channels.is_multiple_of(contraction_part_width) {
        return Err(ProjectionDocumentError::UnevenContractionPartWidth {
            input_channels,
            contraction_part_width,
        });
    }
    // A zero dimension leaves no weights to place, so it is reported as that.
    let Ok(stage_shape) = ProjectionShape::new(input_channels, output_channels, token_count) else {
        return Err(ProjectionDocumentError::WeightCountMismatch {
            expected: 0,
            actual: fp16_weights.len(),
        });
    };
    if fp16_weights.len() != stage_shape.weight_element_count() {
        return Err(ProjectionDocumentError::WeightCountMismatch {
            expected: stage_shape.weight_element_count(),
            actual: fp16_weights.len(),
        });
    }
    Ok(stage_shape)
}

fn part_input_names(stage: &ProjectionStage<'_>) -> Vec<String> {
    if stage.part_count == 1 {
        return vec![stage.input_name.to_owned()];
    }
    let mut part_names = Vec::with_capacity(stage.part_count);
    for part_index in 0..stage.part_count {
        part_names.push(format!("{}_input_{part_index}", stage.stage_name));
    }
    part_names
}

fn stage_split_operations(stage: &ProjectionStage<'_>, part_inputs: &[String]) -> Vec<Vec<u8>> {
    let count_name = format!("{}_split_count", stage.stage_name);
    let axis_name = format!("{}_split_axis", stage.stage_name);
    let part_width = stage.shape.input_channels() / stage.part_count;
    let part_type = activation_type(part_width, stage.shape.token_count());
    let mut split_operation = MilOperation::new("split", &format!("{}_split", stage.stage_name))
        .input_named("x", stage.input_name)
        .input_named("num_splits", &count_name)
        .input_named("axis", &axis_name);
    for part_input in part_inputs {
        split_operation = split_operation.output(part_input, &part_type);
    }
    vec![
        split_count_constant(&count_name, stage.part_count),
        split_axis_constant(&axis_name),
        split_operation.into_bytes(),
    ]
}

fn stage_part_weight(
    stage: &ProjectionStage<'_>,
    part_index: usize,
    weight_name: &str,
    weight_blob_file: &mut WeightBlobFile,
) -> Result<Vec<u8>, ProjectionDocumentError> {
    let part_width = stage.shape.input_channels() / stage.part_count;
    let part_weights = row_block(
        stage.fp16_weights,
        stage.shape.input_channels(),
        part_index * part_width,
        part_width,
    );
    if let StageWeightEncoding::SharedLookupTables { rows_per_table } = stage.weight_encoding {
        let tables = SharedLookupTables::from_fp16_weights(
            &part_weights,
            stage.shape.output_channels(),
            part_width,
            rows_per_table,
        )?;
        return Ok(shared_lookup_table_operation(
            weight_name,
            &tables,
            weight_blob_file,
        ));
    }
    let weight_dimensions = [stage.shape.output_channels(), part_width, 1, 1];
    let blob_offset = weight_blob_file.append_fp16(&part_weights);
    Ok(constant_operation(
        weight_name,
        &weight_blob_value(MilDataType::Float16, &weight_dimensions, blob_offset),
        MilDataType::Float16,
        &weight_dimensions,
    ))
}

pub(crate) fn projection_stage_operations(
    stage: &ProjectionStage<'_>,
    weight_blob_file: &mut WeightBlobFile,
) -> Result<Vec<Vec<u8>>, ProjectionDocumentError> {
    let part_inputs = part_input_names(stage);
    let mut operations = Vec::new();
    if stage.part_count > 1 {
        operations.extend(stage_split_operations(stage, &part_inputs));
    }
    let mut partial_projection_names = Vec::with_capacity(stage.part_count);
    for (part_index, part_input) in part_inputs.iter().enumerate() {
        let weight_name = format!("{}_weight_{part_index}", stage.stage_name);
        operations.push(stage_part_weight(
            stage,
            part_index,
            &weight_name,
            weight_blob_file,
        )?);
        let partial_projection_name = if stage.part_count == 1 {
            stage.output_name.to_owned()
        } else {
            format!("{}_partial_{part_index}", stage.stage_name)
        };
        operations.push(convolution_operation(
            stage.shape,
            part_input,
            &weight_name,
            &partial_projection_name,
        ));
        partial_projection_names.push(partial_projection_name);
    }
    operations.extend(contraction_sum_operations(
        stage.shape,
        &partial_projection_names,
        &format!("{}_sum", stage.stage_name),
        stage.output_name,
    ));
    Ok(operations)
}

/// Splits the fused result, then computes `gate · sigmoid(gate) · up`.
pub(crate) fn silu_gating_operations(
    intermediate_channels: usize,
    token_count: usize,
) -> Vec<Vec<u8>> {
    let intermediate_type = activation_type(intermediate_channels, token_count);
    vec![
        split_count_constant("gate_up_split_count", 2),
        split_axis_constant("gate_up_split_axis"),
        MilOperation::new("split", "gate_up_split")
            .input_named("x", GATE_AND_UP_NAME)
            .input_named("num_splits", "gate_up_split_count")
            .input_named("axis", "gate_up_split_axis")
            .output(GATE_NAME, &intermediate_type)
            .output(UP_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("sigmoid", SIGMOID_NAME)
            .input_named("x", GATE_NAME)
            .output(SIGMOID_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("mul", SILU_NAME)
            .input_named("x", GATE_NAME)
            .input_named("y", SIGMOID_NAME)
            .output(SILU_NAME, &intermediate_type)
            .into_bytes(),
        MilOperation::new("mul", GATED_NAME)
            .input_named("x", SILU_NAME)
            .input_named("y", UP_NAME)
            .output(GATED_NAME, &intermediate_type)
            .into_bytes(),
    ]
}

/// `constexpr_lut_to_dense` over 8-bit codes, with one table per
/// `rows_per_table` rows spanning the whole part.
fn shared_lookup_table_operation(
    weight_name: &str,
    tables: &SharedLookupTables,
    weight_blob_file: &mut WeightBlobFile,
) -> Vec<u8> {
    let weight_dimensions = [tables.output_channels, tables.input_channels, 1, 1];
    let table_dimensions = [
        tables.output_channels / tables.rows_per_table,
        1,
        1,
        1,
        LOOKUP_TABLE_ENTRY_COUNT,
        1,
    ];
    let codes_offset = weight_blob_file.append_uint8(&tables.codes);
    let tables_offset = weight_blob_file.append_fp16(&tables.fp16_tables);
    MilOperation::new("constexpr_lut_to_dense", weight_name)
        .input_value(
            "indices",
            &weight_blob_value(MilDataType::UnsignedInt8, &weight_dimensions, codes_offset),
        )
        .input_value(
            "lut",
            &weight_blob_value(MilDataType::Float16, &table_dimensions, tables_offset),
        )
        .output(
            weight_name,
            &tensor_value_type(MilDataType::Float16, &weight_dimensions),
        )
        .into_bytes()
}
