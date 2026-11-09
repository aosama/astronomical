//! Shape and Metal-template contract shared by ordinary and checkpointing gated-delta kernels.
//!
//! Validation is deliberately independent from dispatch. Both kernels must
//! reject the same incompatible ranks, head geometry, and dtypes before MLX
//! compiles or submits Metal work; otherwise cache-enabled checkpointing could
//! accept a graph that ordinary prompt processing rejects, or vice versa.

use astronomical_runtime_integration::MlxRuntimeError;
use mlx_c_rust::{MlxArray, MlxDtype, MlxMetalKernelTemplateArgument};

const GATED_DELTA_SEQUENCE_OPERATION: &str = "apply fused Qwen3.5 gated-delta sequence";
/// The blocked kernel is written for exactly this key head dimension.
const PIPELINED_KEY_HEAD_DIMENSION: i32 = 128;
/// The kernel partitions each value head into blocks of this many elements.
const VALUE_DIMENSION_BLOCK: i32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GatedDeltaSequenceShape {
    /// Batch rows shared by every sequence input and recurrent state.
    pub(super) batch_size: i32,
    /// Runtime sequence length; intentionally not a Metal template argument.
    pub(super) token_count: i32,
    /// Query/key head count used by the recurrent update.
    pub(super) key_head_count: i32,
    /// Value head count, which must be a multiple of key heads.
    pub(super) value_head_count: i32,
    /// Fixed key dimension required by the blocked kernel implementation.
    pub(super) key_head_dimension: i32,
    /// Value dimension partitioned into 16-element blocks by the kernel.
    pub(super) value_head_dimension: i32,
}

/// Builds only shape-invariant Metal template arguments.
///
/// Token count stays a runtime scalar so different configured prefill chunks
/// and the final partial chunk reuse one compiled pipeline.
pub(super) fn template_arguments(
    sequence_shape: GatedDeltaSequenceShape,
    input_dtype: MlxDtype,
    state_dtype: MlxDtype,
) -> [MlxMetalKernelTemplateArgument; 6] {
    [
        MlxMetalKernelTemplateArgument::Dtype {
            name: "InT",
            dtype: input_dtype,
        },
        MlxMetalKernelTemplateArgument::Dtype {
            name: "StT",
            dtype: state_dtype,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Dk",
            integer_template_argument: sequence_shape.key_head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Dv",
            integer_template_argument: sequence_shape.value_head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Hk",
            integer_template_argument: sequence_shape.key_head_count,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "Hv",
            integer_template_argument: sequence_shape.value_head_count,
        },
    ]
}

pub(super) fn validate_gated_delta_sequence_shapes(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
) -> Result<GatedDeltaSequenceShape, MlxRuntimeError> {
    // Read every shape before validation so error paths never begin native
    // dispatch and both kernel owners observe the same immutable input facts.
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let decay_shape = decays.shape();
    let update_rate_shape = update_rates.shape();
    let recurrent_state_shape = recurrent_state.shape();
    let sequence_shape = sequence_shape_from_input_shapes(&GatedDeltaSequenceInputShapes {
        query_shape: &query_shape,
        key_shape: &key_shape,
        value_shape: &value_shape,
        decay_shape: &decay_shape,
        update_rate_shape: &update_rate_shape,
        recurrent_state_shape: &recurrent_state_shape,
    })?;
    validate_sequence_dtypes(
        recurrent_state.dtype(),
        [
            queries.dtype(),
            keys.dtype(),
            values.dtype(),
            decays.dtype(),
            update_rates.dtype(),
        ],
    )?;
    Ok(sequence_shape)
}

/// The shapes of the six kernel inputs, read once from their arrays.
pub(super) struct GatedDeltaSequenceInputShapes<'shape> {
    pub(super) query_shape: &'shape [i32],
    pub(super) key_shape: &'shape [i32],
    pub(super) value_shape: &'shape [i32],
    pub(super) decay_shape: &'shape [i32],
    pub(super) update_rate_shape: &'shape [i32],
    pub(super) recurrent_state_shape: &'shape [i32],
}

/// Derives the kernel geometry from the queries and values, then requires
/// every other input to agree with it.
pub(super) fn sequence_shape_from_input_shapes(
    input_shapes: &GatedDeltaSequenceInputShapes<'_>,
) -> Result<GatedDeltaSequenceShape, MlxRuntimeError> {
    let has_expected_ranks = input_shapes.query_shape.len() == 4
        && input_shapes.key_shape.len() == 4
        && input_shapes.value_shape.len() == 4
        && input_shapes.decay_shape.len() == 3
        && input_shapes.update_rate_shape.len() == 3
        && input_shapes.recurrent_state_shape.len() == 4;
    if !has_expected_ranks {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence inputs have invalid ranks",
        ));
    }
    if input_shapes.query_shape != input_shapes.key_shape {
        return Err(gated_delta_sequence_error(
            "fused gated-delta queries and keys must have identical shapes",
        ));
    }
    let sequence_shape = GatedDeltaSequenceShape {
        batch_size: input_shapes.query_shape[0],
        token_count: input_shapes.query_shape[1],
        key_head_count: input_shapes.query_shape[2],
        key_head_dimension: input_shapes.query_shape[3],
        value_head_count: input_shapes.value_shape[2],
        value_head_dimension: input_shapes.value_shape[3],
    };
    sequence_shape.validate_kernel_geometry()?;
    if !sequence_shape.matches_companion_shapes(input_shapes) {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence shapes are incompatible",
        ));
    }
    Ok(sequence_shape)
}

impl GatedDeltaSequenceShape {
    /// The blocked kernel's own limits, each reported on its own so a model
    /// that breaks one learns which.
    fn validate_kernel_geometry(&self) -> Result<(), MlxRuntimeError> {
        let dimensions = [
            self.batch_size,
            self.token_count,
            self.key_head_count,
            self.value_head_count,
            self.key_head_dimension,
            self.value_head_dimension,
        ];
        if dimensions.into_iter().min() <= Some(0) {
            return Err(gated_delta_sequence_error(
                "pipelined gated-delta dimensions must be positive",
            ));
        }
        if self.value_head_count % self.key_head_count != 0 {
            return Err(gated_delta_sequence_error(
                "pipelined gated-delta value heads must be a multiple of key heads",
            ));
        }
        if self.key_head_dimension != PIPELINED_KEY_HEAD_DIMENSION {
            return Err(gated_delta_sequence_error(
                "pipelined gated-delta key dimension must be 128",
            ));
        }
        if self.value_head_dimension % VALUE_DIMENSION_BLOCK != 0 {
            return Err(gated_delta_sequence_error(
                "pipelined gated-delta value dimension must divide by 16",
            ));
        }
        Ok(())
    }

    /// Values share batch and tokens with the queries; decays and update
    /// rates are per value head; the state is one `[Dv, Dk]` matrix per head.
    fn matches_companion_shapes(&self, input_shapes: &GatedDeltaSequenceInputShapes<'_>) -> bool {
        let per_value_head_shape = [self.batch_size, self.token_count, self.value_head_count];
        let recurrent_state_shape = [
            self.batch_size,
            self.value_head_count,
            self.value_head_dimension,
            self.key_head_dimension,
        ];
        input_shapes.value_shape[0] == self.batch_size
            && input_shapes.value_shape[1] == self.token_count
            && input_shapes.decay_shape == per_value_head_shape
            && input_shapes.update_rate_shape == per_value_head_shape
            && input_shapes.recurrent_state_shape == recurrent_state_shape
    }
}

/// The state accumulates across tokens in float32 whatever the activation
/// dtype, so a narrower state would drift over a long prompt.
pub(super) fn validate_sequence_dtypes(
    recurrent_state_dtype: MlxDtype,
    activation_dtypes: [MlxDtype; 5],
) -> Result<(), MlxRuntimeError> {
    if recurrent_state_dtype != MlxDtype::Float32 {
        return Err(gated_delta_sequence_error(
            "fused gated-delta recurrent state must use float32",
        ));
    }
    if !activation_dtypes
        .into_iter()
        .all(is_supported_activation_dtype)
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta inputs must use float16, bfloat16, or float32",
        ));
    }
    Ok(())
}

fn is_supported_activation_dtype(dtype: MlxDtype) -> bool {
    matches!(
        dtype,
        MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
    )
}

pub(super) fn gated_delta_sequence_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: GATED_DELTA_SEQUENCE_OPERATION,
        description: description.to_owned(),
    }
}

#[cfg(test)]
#[path = "gated_delta_sequence_contract_tests.rs"]
mod gated_delta_sequence_contract_tests;
