//! Fused Qwen3.5 gated-delta decode prework kernel.
//!
//! One Metal launch replaces the composed decode-side chain between the
//! projections and the recurrence: the rolling convolution window, the
//! depthwise conv1d, SiLU, the q/k/v split, the two ones-weight RMS
//! normalizations, the two scalar scales, and the next rolling-state slice —
//! roughly thirteen dispatches per gated-delta layer that are kernel-launch
//! bound at one-token decode shapes.
//!
//! The Metal source, with the numerics notes that tie it bit for bit to the
//! composed production path, is
//! `metal/astro_family_qwen35_gated_delta_decode_prework.metal`.
//!
//! The kernel geometry is fixed to the Qwen3.5 gated-delta layout: the head
//! dimension must equal the thirty-two-lane times four-element window, and
//! the rolling buffer keeps `kernel_dimension - 1` rows.
//!
//! The fused path engages only for bfloat16 activations: bfloat16 has no
//! native Metal arithmetic, so both the fused launch and MLX's composed path
//! round through float32 identically, and the direct-MLX numerics contract
//! proves bit-for-bit parity. float16 uses native half arithmetic whose
//! `exp` intrinsic a separately-compiled launch does not reproduce bit-for-bit,
//! so float16 falls back to the composed path rather than risk diverging bits.

use crate::metal_kernels::ASTRO_FAMILY_QWEN35_GATED_DELTA_DECODE_PREWORK;
use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use model_primitives::metal_kernel_declaration::{declare_metal_kernel, float_define};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

/// Rolling convolution rows retained across steps: conv kernel size minus one.
const KEPT_STATE_ROW_COUNT: i32 = 3;
/// Thirty-two lanes each own four contiguous head channels.
pub const LANE_COUNT: i32 = 32;
const CHANNELS_PER_LANE: i32 = 4;
/// The only head dimension this kernel serves: `LANE_COUNT * CHANNELS_PER_LANE`.
const SUPPORTED_HEAD_DIMENSION: i32 = LANE_COUNT * CHANNELS_PER_LANE;
/// The only activation dtype with verified bit-exact parity. bfloat16 has no
/// native Metal arithmetic (it emulates through float32), so the fused kernel
/// and MLX's composed path round identically. float16 has native half
/// arithmetic, and Metal's `exp(half)` is a distinct lower-precision intrinsic
/// that a separately-compiled fused launch does not reproduce bit-for-bit;
/// float16 therefore falls back to the composed path.
const SUPPORTED_ACTIVATION_DTYPES: [MlxDtype; 1] = [MlxDtype::BFloat16];
/// Launch-bound shapes only: single decode steps and verification rows.
const MAX_FUSED_TOKEN_COUNT: i32 = KEPT_STATE_ROW_COUNT + 1;

const PREWORK_OPERATION: &str = "apply the fused Qwen3.5 gated-delta decode prework";

/// One fused prework launch result: the normalized, scaled q/k/v heads the
/// recurrence consumes plus the rolling convolution state for the next step.
pub struct GdnDecodePreworkOutput {
    pub queries: MlxArray,
    pub keys: MlxArray,
    pub values: MlxArray,
    pub next_convolution_state: MlxArray,
}

/// Builds the fused decode prework kernel
/// (`metal/astro_family_qwen35_gated_delta_decode_prework.metal`) with the
/// model's RMS epsilon baked into the source: Metal kernel template arguments
/// carry no floats.
pub fn qwen3_5_gdn_decode_prework_kernel(
    rms_norm_epsilon: f32,
) -> Result<MlxMetalKernel, runtime_integration::MlxRuntimeError> {
    declare_metal_kernel(
        &ASTRO_FAMILY_QWEN35_GATED_DELTA_DECODE_PREWORK,
        &[float_define("RMS_NORM_EPSILON", rms_norm_epsilon)],
    )
}

/// Applies the fused decode prework. The token count comes from the mixed
/// q/k/v batch shape, so the caller cannot desynchronize it from the data.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gdn_decode_prework(
    runtime: &MlxRuntime,
    prework_kernel: &MlxMetalKernel,
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
    mixed_queries_keys_values: &MlxArray,
    convolution_state: &MlxArray,
    convolution_weight: &MlxArray,
    query_scale: &MlxArray,
    key_scale: &MlxArray,
) -> Result<GdnDecodePreworkOutput, MlxRuntimeError> {
    let shape = validate_prework_shapes(
        key_head_count,
        value_head_count,
        head_dimension,
        mixed_queries_keys_values,
        convolution_state,
        convolution_weight,
        query_scale,
        key_scale,
    )?;
    let activation_dtype = mixed_queries_keys_values.dtype();
    let [
        queries_shape,
        keys_shape,
        values_shape,
        next_convolution_state_shape,
    ] = prework_output_shapes(&shape, key_head_count, value_head_count, head_dimension);
    let outputs = runtime.apply_metal_kernel(
        prework_kernel,
        &[
            mixed_queries_keys_values,
            convolution_state,
            convolution_weight,
            query_scale,
            key_scale,
        ],
        &[
            MlxMetalKernelOutput::new(queries_shape, activation_dtype),
            MlxMetalKernelOutput::new(keys_shape, activation_dtype),
            MlxMetalKernelOutput::new(values_shape, activation_dtype),
            MlxMetalKernelOutput::new(next_convolution_state_shape, activation_dtype),
        ],
        prework_launch_grid(&shape, key_head_count, value_head_count),
        [LANE_COUNT, 1, 1],
        &prework_template_arguments(
            key_head_count,
            value_head_count,
            head_dimension,
            shape.convolution_dimension,
            shape.token_count,
            activation_dtype,
        ),
    )?;
    unpack_prework_outputs(outputs)
}

/// Shapes of the kernel's four outputs in the order it declares them: the
/// normalized queries, the normalized keys, the split values, and the next
/// rolling convolution state.
fn prework_output_shapes(
    shape: &PreworkShape,
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
) -> [Vec<i32>; 4] {
    [
        vec![
            shape.batch_size,
            shape.token_count,
            key_head_count,
            head_dimension,
        ],
        vec![
            shape.batch_size,
            shape.token_count,
            key_head_count,
            head_dimension,
        ],
        vec![
            shape.batch_size,
            shape.token_count,
            value_head_count,
            head_dimension,
        ],
        vec![
            shape.batch_size,
            KEPT_STATE_ROW_COUNT,
            shape.convolution_dimension,
        ],
    ]
}

/// One threadgroup per (token row, logical head), where the logical heads are
/// the query heads, the key heads, then the value heads.
fn prework_launch_grid(
    shape: &PreworkShape,
    key_head_count: i32,
    value_head_count: i32,
) -> [i32; 3] {
    [
        LANE_COUNT,
        shape.batch_size * shape.token_count,
        2 * key_head_count + value_head_count,
    ]
}

/// Takes the four launch results in kernel-declared order, naming the first
/// one a short return left out.
fn unpack_prework_outputs(
    outputs: Vec<MlxArray>,
) -> Result<GdnDecodePreworkOutput, MlxRuntimeError> {
    let mut output_iterator = outputs.into_iter();
    let queries = output_iterator.next().ok_or_else(|| {
        prework_error("the fused prework kernel did not return normalized queries")
    })?;
    let keys = output_iterator
        .next()
        .ok_or_else(|| prework_error("the fused prework kernel did not return normalized keys"))?;
    let values = output_iterator
        .next()
        .ok_or_else(|| prework_error("the fused prework kernel did not return split values"))?;
    let next_convolution_state = output_iterator.next().ok_or_else(|| {
        prework_error("the fused prework kernel did not return the next convolution state")
    })?;
    Ok(GdnDecodePreworkOutput {
        queries,
        keys,
        values,
        next_convolution_state,
    })
}

/// True when the composed path should hand this step to the fused kernel:
/// launch-bound token counts, the verified activation dtype, and the exact
/// head geometry the kernel's lane window serves.
pub fn is_gdn_decode_prework_eligible(
    prework_kernel: Option<&MlxMetalKernel>,
    token_count: i32,
    activation_dtype: MlxDtype,
    head_dimension: i32,
) -> bool {
    prework_kernel.is_some()
        && (1..=MAX_FUSED_TOKEN_COUNT).contains(&token_count)
        && SUPPORTED_ACTIVATION_DTYPES.contains(&activation_dtype)
        && head_dimension == SUPPORTED_HEAD_DIMENSION
}

struct PreworkShape {
    batch_size: i32,
    token_count: i32,
    convolution_dimension: i32,
}

#[allow(clippy::too_many_arguments)]
fn validate_prework_shapes(
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
    mixed_queries_keys_values: &MlxArray,
    convolution_state: &MlxArray,
    convolution_weight: &MlxArray,
    query_scale: &MlxArray,
    key_scale: &MlxArray,
) -> Result<PreworkShape, MlxRuntimeError> {
    if head_dimension != SUPPORTED_HEAD_DIMENSION {
        return Err(prework_error(
            "the fused prework kernel serves only the thirty-two-lane four-channel head window",
        ));
    }
    if key_head_count <= 0 || value_head_count <= 0 {
        return Err(prework_error("head counts must be positive"));
    }
    let mixed_shape = mixed_queries_keys_values.shape();
    let state_shape = convolution_state.shape();
    let weight_shape = convolution_weight.shape();
    if mixed_shape.len() != 3
        || state_shape.len() != 3
        || weight_shape.len() != 3
        // MLX hands a 0-dimensional array to the kernel by value, which is how
        // the kernel reads the scales; a one-element vector would arrive as a
        // pointer and the kernel would not compile.
        || !query_scale.shape().is_empty()
        || !key_scale.shape().is_empty()
    {
        return Err(prework_error(
            "prework operands must use the production ranks: rank-three qkv, state, and weight; scalar scales",
        ));
    }
    let batch_size = mixed_shape[0];
    let token_count = mixed_shape[1];
    let convolution_dimension = mixed_shape[2];
    let expected_convolution_dimension =
        2 * key_head_count * head_dimension + value_head_count * head_dimension;
    if batch_size != 1 {
        return Err(prework_error(
            "the fused prework kernel serves single-batch decode steps",
        ));
    }
    if !(1..=MAX_FUSED_TOKEN_COUNT).contains(&token_count) {
        return Err(prework_error(
            "the fused prework kernel serves single decode steps and verification rows",
        ));
    }
    if convolution_dimension != expected_convolution_dimension {
        return Err(prework_error(
            "the qkv channel count must equal two key head blocks plus the value head block",
        ));
    }
    if state_shape != [batch_size, KEPT_STATE_ROW_COUNT, convolution_dimension] {
        return Err(prework_error(
            "the convolution state must keep exactly three rolling rows of the channel layout",
        ));
    }
    if weight_shape != [convolution_dimension, KEPT_STATE_ROW_COUNT + 1, 1] {
        return Err(prework_error(
            "the convolution weight must be per-channel four-tap depthwise",
        ));
    }
    let activation_dtype = mixed_queries_keys_values.dtype();
    if !SUPPORTED_ACTIVATION_DTYPES.contains(&activation_dtype)
        || convolution_state.dtype() != activation_dtype
        || convolution_weight.dtype() != activation_dtype
    {
        return Err(prework_error(
            "prework operands must share the bfloat16 activation dtype",
        ));
    }
    Ok(PreworkShape {
        batch_size,
        token_count,
        convolution_dimension,
    })
}

fn prework_template_arguments(
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
    convolution_dimension: i32,
    token_count: i32,
    activation_dtype: MlxDtype,
) -> Vec<MlxMetalKernelTemplateArgument> {
    vec![
        MlxMetalKernelTemplateArgument::Dtype {
            name: "T",
            dtype: activation_dtype,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "HK",
            integer_template_argument: key_head_count,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "HV",
            integer_template_argument: value_head_count,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "DK",
            integer_template_argument: head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "DV",
            integer_template_argument: head_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "NKEEP",
            integer_template_argument: KEPT_STATE_ROW_COUNT,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "C",
            integer_template_argument: convolution_dimension,
        },
        MlxMetalKernelTemplateArgument::Integer {
            name: "S",
            integer_template_argument: token_count,
        },
    ]
}

fn prework_error(description: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: PREWORK_OPERATION,
        description: description.to_owned(),
    }
}

#[cfg(test)]
#[path = "gdn_decode_prework_kernel_tests.rs"]
mod tests;
