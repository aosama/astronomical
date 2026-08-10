//! Shape and Metal-template contract shared by ordinary and checkpointing gated-delta kernels.
//!
//! Validation is deliberately independent from dispatch. Both kernels must
//! reject the same incompatible ranks, head geometry, and dtypes before MLX
//! compiles or submits Metal work; otherwise cache-enabled checkpointing could
//! accept a graph that ordinary prompt processing rejects, or vice versa.

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMetalKernelTemplateArgument, MlxRuntimeError,
};

const GATED_DELTA_SEQUENCE_OPERATION: &str = "apply fused Qwen3.5 gated-delta sequence";

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
    /// Value dimension partitioned into 32-element blocks by the kernel.
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
    if query_shape.len() != 4
        || key_shape.len() != 4
        || value_shape.len() != 4
        || decay_shape.len() != 3
        || update_rate_shape.len() != 3
        || recurrent_state_shape.len() != 4
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence inputs have invalid ranks",
        ));
    }
    if query_shape != key_shape {
        return Err(gated_delta_sequence_error(
            "fused gated-delta queries and keys must have identical shapes",
        ));
    }
    let batch_size = query_shape[0];
    let token_count = query_shape[1];
    let key_head_count = query_shape[2];
    let key_head_dimension = query_shape[3];
    let value_head_count = value_shape[2];
    let value_head_dimension = value_shape[3];
    if batch_size <= 0
        || token_count <= 0
        || key_head_count <= 0
        || value_head_count <= 0
        || key_head_dimension <= 0
        || value_head_dimension <= 0
        || value_head_count % key_head_count != 0
        || key_head_dimension != 128
        || value_head_dimension % 32 != 0
    {
        return Err(gated_delta_sequence_error(
            "blocked gated-delta dimensions must be positive, value heads must be a multiple of key heads, key dimension must be 128, and value dimension must divide by 32",
        ));
    }
    if value_shape[0] != batch_size
        || value_shape[1] != token_count
        || decay_shape != [batch_size, token_count, value_head_count]
        || update_rate_shape != decay_shape
        || recurrent_state_shape
            != [
                batch_size,
                value_head_count,
                value_head_dimension,
                key_head_dimension,
            ]
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta sequence shapes are incompatible",
        ));
    }
    if recurrent_state.dtype() != MlxDtype::Float32 {
        return Err(gated_delta_sequence_error(
            "fused gated-delta recurrent state must use float32",
        ));
    }
    if !is_supported_activation_dtype(queries.dtype())
        || !is_supported_activation_dtype(keys.dtype())
        || !is_supported_activation_dtype(values.dtype())
        || !is_supported_activation_dtype(decays.dtype())
        || !is_supported_activation_dtype(update_rates.dtype())
    {
        return Err(gated_delta_sequence_error(
            "fused gated-delta inputs must use float16, bfloat16, or float32",
        ));
    }
    Ok(GatedDeltaSequenceShape {
        batch_size,
        token_count,
        key_head_count,
        value_head_count,
        key_head_dimension,
        value_head_dimension,
    })
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
