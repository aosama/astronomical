//! Fused Qwen3.5 gated-delta decode prework kernel.
//!
//! One Metal launch replaces the composed decode-side chain between the
//! projections and the recurrence: the rolling convolution window, the
//! depthwise conv1d, SiLU, the q/k/v split, the two ones-weight RMS
//! normalizations, the two scalar scales, and the next rolling-state slice —
//! roughly thirteen dispatches per gated-delta layer that are kernel-launch
//! bound at one-token decode shapes.
//!
//! Numerics mirror the composed production path exactly, kernel by kernel:
//! the conv accumulates four taps in float32 and rounds once like MLX's
//! implicit-GEMM depthwise conv; the in-kernel SiLU uses MLX's own unary
//! sigmoid formula (exp-of-abs), swept bit-exact over all finite inputs by
//! the donor kernel; the RMS reduction follows `rms_single_row` from MLX's
//! Metal kernels — contiguous four-element lanes, one simd-group sum, and
//! `rsqrt(sum_of_squares / head_dimension + epsilon)` with this model's
//! configured epsilon — followed by the ones-weight rounding cast and the
//! separate scalar-scale rounding. The direct-MLX numerics contract proves
//! bit-for-bit parity before any dispatch engages.
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

use astronomical_mlx_c_rust::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

/// Rolling convolution rows retained across steps: conv kernel size minus one.
const KEPT_STATE_ROW_COUNT: i32 = 3;
/// Thirty-two lanes each own four contiguous head channels.
pub(crate) const LANE_COUNT: i32 = 32;
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

/// Builds the fused decode prework kernel with the model's RMS epsilon baked
/// into the source as a literal — Metal kernel template arguments carry no
/// floats.
pub fn qwen3_5_gdn_decode_prework_kernel(
    rms_norm_epsilon: f32,
) -> Result<MlxMetalKernel, astronomical_runtime_integration::MlxRuntimeError> {
    MlxMetalKernel::new(
        "astronomical_qwen3_5_gdn_decode_prework",
        &[
            "mixed_queries_keys_values",
            "convolution_state",
            "convolution_weight",
            "query_scale",
            "key_scale",
        ],
        &["queries", "keys", "values", "next_convolution_state"],
        &gdn_decode_prework_kernel_source(rms_norm_epsilon),
    )
    .map_err(astronomical_runtime_integration::MlxRuntimeError::from)
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
            MlxMetalKernelOutput::new(
                vec![
                    shape.batch_size,
                    shape.token_count,
                    key_head_count,
                    head_dimension,
                ],
                mixed_queries_keys_values.dtype(),
            ),
            MlxMetalKernelOutput::new(
                vec![
                    shape.batch_size,
                    shape.token_count,
                    key_head_count,
                    head_dimension,
                ],
                mixed_queries_keys_values.dtype(),
            ),
            MlxMetalKernelOutput::new(
                vec![
                    shape.batch_size,
                    shape.token_count,
                    value_head_count,
                    head_dimension,
                ],
                mixed_queries_keys_values.dtype(),
            ),
            MlxMetalKernelOutput::new(
                vec![
                    shape.batch_size,
                    KEPT_STATE_ROW_COUNT,
                    shape.convolution_dimension,
                ],
                mixed_queries_keys_values.dtype(),
            ),
        ],
        [
            LANE_COUNT,
            shape.batch_size * shape.token_count,
            2 * key_head_count + value_head_count,
        ],
        [LANE_COUNT, 1, 1],
        &prework_template_arguments(
            key_head_count,
            value_head_count,
            head_dimension,
            shape.convolution_dimension,
            shape.token_count,
            mixed_queries_keys_values.dtype(),
        ),
    )?;
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
        && token_count >= 1
        && token_count <= MAX_FUSED_TOKEN_COUNT
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
        || query_scale.shape().len() > 1
        || key_scale.shape().len() > 1
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
    if token_count < 1 || token_count > MAX_FUSED_TOKEN_COUNT {
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

/// Metal source for the fused prework, adapted from an upstream open-source
/// Qwen3.5 gated-delta kernel (see third-party/THIRD_PARTY_NOTICES for the
/// provenance and license). The RMS epsilon placement follows this
/// repository's `rms_norm_without_weight` reference
/// (`rsqrt(sum_of_squares / head_dimension + epsilon)`), not the upstream
/// source's.
fn gdn_decode_prework_kernel_source(rms_norm_epsilon: f32) -> String {
    let epsilon_literal = format!("{rms_norm_epsilon:.9e}");
    format!(
        r#"
    uint lane = thread_position_in_threadgroup.x;
    uint batch_index = threadgroup_position_in_grid.y / uint(S);
    uint row = threadgroup_position_in_grid.y % uint(S);
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sum_of_squares = 0.0f;
    for (uint element_index = 0; element_index < 4; ++element_index) {{
        uint channel = channel_base + lane * 4 + element_index;
        float tap_accumulator = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {{
            uint input_row = row + tap;
            const T window_value = input_row < uint(NKEEP)
                ? convolution_state[(batch_index * uint(NKEEP) + input_row) * uint(C) + channel]
                : mixed_queries_keys_values[(batch_index * uint(S) + input_row - uint(NKEEP)) * uint(C) + channel];
            tap_accumulator += float(window_value) * float(convolution_weight[channel * 4 + tap]);
        }}
        const T convolved = T(tap_accumulator);
        T sigmoid_branch = T(1) / (T(1) + metal::exp(metal::abs(convolved)));
        const T silu_value = convolved * ((convolved < T(0)) ? sigmoid_branch : T(1) - sigmoid_branch);
        activated[element_index] = silu_value;
        float activated_value = float(silu_value);
        sum_of_squares += activated_value * activated_value;
    }}
    if (is_q || is_k) {{
        sum_of_squares = simd_sum(sum_of_squares);
        float inverse_mean = metal::precise::rsqrt(sum_of_squares / float(DK) + {epsilon_literal}f);
        const T scale = is_q ? query_scale : key_scale;
        uint out_base = ((batch_index * uint(S) + row) * uint(HK) + head) * uint(DK) + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {{
            const T normalized = T(1) * T(float(activated[element_index]) * inverse_mean);
            const T scaled = scale * normalized;
            if (is_q) {{
                queries[out_base + element_index] = scaled;
            }} else {{
                keys[out_base + element_index] = scaled;
            }}
        }}
    }} else {{
        uint out_base = ((batch_index * uint(S) + row) * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {{
            values[out_base + element_index] = activated[element_index];
        }}
    }}
    if (S < NKEEP && row == 0) {{
        for (uint retained_row = 0; retained_row < uint(NKEEP) - uint(S); ++retained_row) {{
            uint destination = (batch_index * uint(NKEEP) + retained_row) * uint(C)
                               + channel_base + lane * 4;
            uint source = (batch_index * uint(NKEEP) + retained_row + uint(S)) * uint(C)
                          + channel_base + lane * 4;
            for (uint element_index = 0; element_index < 4; ++element_index) {{
                next_convolution_state[destination + element_index] = convolution_state[source + element_index];
            }}
        }}
    }}
    if (row + uint(NKEEP) >= uint(S)) {{
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = (batch_index * uint(S) + row) * uint(C) + channel_base + lane * 4;
        uint state_base = (batch_index * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
        for (uint element_index = 0; element_index < 4; ++element_index) {{
            next_convolution_state[state_base + element_index] = mixed_queries_keys_values[raw_base + element_index];
        }}
    }}
"#
    )
}
