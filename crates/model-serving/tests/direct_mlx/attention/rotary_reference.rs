use astronomical_model_serving::{
    compute_default_rope_frequency_denominators, compute_yarn_rope_frequency_denominators,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[derive(Clone, Copy)]
pub(crate) struct RotaryGeometry {
    pub(crate) row_name: &'static str,
    pub(crate) head_width: i32,
    pub(crate) rotary_dimension: i32,
    pub(crate) activation_dtype: MlxDtype,
    pub(crate) token_positions: &'static [i32],
    pub(crate) attention_factor: f32,
    pub(crate) frequency_kind: FrequencyKind,
}

#[derive(Clone, Copy)]
pub(crate) enum FrequencyKind {
    Default {
        theta: f64,
    },
    Yarn {
        theta: f64,
        original_maximum_position_count: u32,
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
    },
}

#[tokio::test]
async fn should_match_operations_reference_for_generic_rotary_rows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let rows = [
        RotaryGeometry {
            row_name: "generic_default_partial",
            head_width: 8,
            rotary_dimension: 4,
            activation_dtype: MlxDtype::Float32,
            token_positions: &[0, 3],
            attention_factor: 1.0,
            frequency_kind: FrequencyKind::Default { theta: 10_000.0 },
        },
        RotaryGeometry {
            row_name: "generic_yarn_partial",
            head_width: 16,
            rotary_dimension: 8,
            activation_dtype: MlxDtype::BFloat16,
            token_positions: &[2, 7, 13],
            attention_factor: 1.2,
            frequency_kind: FrequencyKind::Yarn {
                theta: 500_000.0,
                original_maximum_position_count: 8_192,
                factor: 4.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        },
    ];

    for geometry in rows {
        assert_rotary_matches_operations_reference(&runtime, geometry);
    }
}

#[tokio::test]
async fn should_match_softplus_gate_reference_without_widening_activation_dtype() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let rows = [
        (
            "float32_per_head",
            MlxDtype::Float32,
            vec![1, 2, 2, 4],
            vec![1, 2, 2, 1],
        ),
        (
            "float16_per_token",
            MlxDtype::Float16,
            vec![1, 2, 3, 4],
            vec![1, 1, 3, 1],
        ),
        (
            "bfloat16_per_element",
            MlxDtype::BFloat16,
            vec![1, 2, 2, 4],
            vec![1, 2, 2, 4],
        ),
    ];

    for (row_name, activation_dtype, output_shape, gate_shape) in rows {
        let attention_output = deterministic_array(&runtime, &output_shape, activation_dtype, 17);
        let gate_logits = deterministic_array(&runtime, &gate_shape, activation_dtype, 11);
        let gated_output = runtime
            .apply_softplus_attention_gate(&attention_output, &gate_logits)
            .unwrap_or_else(|error| panic!("{row_name} gate should execute: {error}"));
        assert_eq!(gated_output.dtype(), activation_dtype, "{row_name}");

        let output_values = float32_values(&runtime, &attention_output, row_name);
        let gate_values = float32_values(&runtime, &gate_logits, row_name);
        let expected = broadcast_softplus_gate_reference(
            &output_values,
            &gate_values,
            &output_shape,
            &gate_shape,
        );
        let actual = float32_values(&runtime, &gated_output, row_name);
        let tolerance = if activation_dtype == MlxDtype::Float32 {
            2e-5
        } else {
            2e-2
        };
        assert_values_close(row_name, &actual, &expected, tolerance);
    }
}

#[tokio::test]
async fn should_reject_non_broadcast_attention_gate_geometry() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let output = deterministic_array(&runtime, &[1, 2, 3, 4], MlxDtype::Float32, 13);
    let invalid_gate = deterministic_array(&runtime, &[1, 2, 2, 1], MlxDtype::Float32, 7);
    runtime
        .apply_softplus_attention_gate(&output, &invalid_gate)
        .expect_err("a gate whose token axis cannot broadcast must fail before execution");
}

pub(crate) fn assert_rotary_matches_operations_reference(
    runtime: &MlxRuntime,
    geometry: RotaryGeometry,
) {
    let token_count = geometry.token_positions.len() as i32;
    let mut source_values = deterministic_values(2 * token_count * geometry.head_width, 23);
    // The caller applies its descriptor attention factor only to the rotary
    // prefix. Preparing that prefix explicitly proves the untouched tail too.
    for head_index in 0..2_usize {
        for token_index in 0..token_count as usize {
            for dimension_index in 0..geometry.rotary_dimension as usize {
                let element_index = (head_index * token_count as usize + token_index)
                    * geometry.head_width as usize
                    + dimension_index;
                source_values[element_index] *= geometry.attention_factor;
            }
        }
    }
    let prepared_input = runtime
        .array_from_f32(&source_values, &[1, 2, token_count, geometry.head_width])
        .and_then(|array| runtime.astype(&array, geometry.activation_dtype))
        .unwrap_or_else(|error| panic!("{} input should build: {error}", geometry.row_name));
    let frequency_denominators = frequency_denominators(geometry);
    let denominator_array = runtime
        .array_from_f32(&frequency_denominators, &[geometry.rotary_dimension / 2])
        .expect("frequency denominator array should build");
    let token_positions = runtime
        .array_from_i32(geometry.token_positions, &[token_count])
        .expect("token positions should build");
    let rotated = runtime
        .rope_with_custom_frequencies_at_positions(
            &prepared_input,
            &token_positions,
            geometry.rotary_dimension,
            &denominator_array,
            1.0,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{} rotary execution should build: {error}",
                geometry.row_name
            )
        });
    assert_eq!(
        rotated.dtype(),
        geometry.activation_dtype,
        "{}",
        geometry.row_name
    );

    let evaluated_input = float32_values(runtime, &prepared_input, geometry.row_name);
    let expected = rotary_operations_reference(&evaluated_input, geometry, &frequency_denominators);
    let actual = float32_values(runtime, &rotated, geometry.row_name);
    let tolerance = if geometry.activation_dtype == MlxDtype::Float32 {
        2e-5
    } else {
        3e-2
    };
    assert_values_close(geometry.row_name, &actual, &expected, tolerance);
}

fn frequency_denominators(geometry: RotaryGeometry) -> Vec<f32> {
    match geometry.frequency_kind {
        FrequencyKind::Default { theta } => {
            compute_default_rope_frequency_denominators(theta, geometry.rotary_dimension as u32)
                .expect("default frequencies should be valid")
        }
        FrequencyKind::Yarn {
            theta,
            original_maximum_position_count,
            factor,
            beta_fast,
            beta_slow,
        } => compute_yarn_rope_frequency_denominators(
            theta,
            geometry.rotary_dimension as u32,
            original_maximum_position_count,
            factor,
            beta_fast,
            beta_slow,
        )
        .expect("YaRN frequencies should be valid")
        .frequency_denominators()
        .to_vec(),
    }
}

fn rotary_operations_reference(
    prepared_input: &[f32],
    geometry: RotaryGeometry,
    frequency_denominators: &[f32],
) -> Vec<f32> {
    let token_count = geometry.token_positions.len();
    let head_width = geometry.head_width as usize;
    let rotary_dimension = geometry.rotary_dimension as usize;
    let rotary_half = rotary_dimension / 2;
    let mut output = prepared_input.to_vec();
    for head_index in 0..2_usize {
        for token_index in 0..token_count {
            let token_base = (head_index * token_count + token_index) * head_width;
            let token_position = geometry.token_positions[token_index] as f32;
            for pair_index in 0..rotary_half {
                let angle = token_position / frequency_denominators[pair_index];
                let cosine = angle.cos();
                let sine = angle.sin();
                let first_index = token_base + pair_index;
                let second_index = token_base + rotary_half + pair_index;
                let first_value = prepared_input[first_index];
                let second_value = prepared_input[second_index];
                output[first_index] = first_value * cosine - second_value * sine;
                output[second_index] = first_value * sine + second_value * cosine;
            }
        }
    }
    output
}

fn broadcast_softplus_gate_reference(
    output_values: &[f32],
    gate_values: &[f32],
    output_shape: &[i32],
    gate_shape: &[i32],
) -> Vec<f32> {
    let output_strides = row_major_strides(output_shape);
    let gate_strides = row_major_strides(gate_shape);
    let rank_offset = output_shape.len() - gate_shape.len();
    (0..output_values.len())
        .map(|output_element_index| {
            let mut remaining_index = output_element_index;
            let mut gate_element_index = 0_usize;
            for (output_axis_index, output_stride) in output_strides.iter().enumerate() {
                let coordinate = remaining_index / output_stride;
                remaining_index %= output_stride;
                if output_axis_index >= rank_offset {
                    let gate_axis_index = output_axis_index - rank_offset;
                    let gate_coordinate = if gate_shape[gate_axis_index] == 1 {
                        0
                    } else {
                        coordinate
                    };
                    gate_element_index += gate_coordinate * gate_strides[gate_axis_index];
                }
            }
            output_values[output_element_index] * softplus(gate_values[gate_element_index])
        })
        .collect()
}

fn row_major_strides(shape: &[i32]) -> Vec<usize> {
    let mut strides = vec![1_usize; shape.len()];
    for axis_index in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis_index] = strides[axis_index + 1] * shape[axis_index + 1] as usize;
    }
    strides
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn deterministic_array(
    runtime: &MlxRuntime,
    shape: &[i32],
    dtype: MlxDtype,
    period: usize,
) -> MlxArray {
    let values = deterministic_values(shape.iter().product(), period);
    runtime
        .array_from_f32(&values, shape)
        .and_then(|array| runtime.astype(&array, dtype))
        .expect("deterministic array should build")
}

fn deterministic_values(element_count: i32, period: usize) -> Vec<f32> {
    let midpoint = (period / 2) as f32;
    (0..element_count as usize)
        .map(|element_index| ((element_index % period) as f32 - midpoint) / 16.0)
        .collect()
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray, row_name: &str) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| runtime.build_contiguous_row_major_copy(&float32_array))
        .and_then(|contiguous_array| contiguous_array.to_vec_f32())
        .unwrap_or_else(|error| panic!("{row_name} should evaluate: {error}"))
}

fn assert_values_close(row_name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{row_name}");
    for (element_index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= tolerance * comparison_scale,
            "{row_name} element {element_index}: expected {expected_value}, got {actual_value}"
        );
    }
}

pub(crate) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("rotary-reference test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
