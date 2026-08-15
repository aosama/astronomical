//! Deterministic MLX fixtures shared by the resident gate/up acceptance test.

use astronomical_model_serving::qwen3_5_moe_sort_expert_assignments;
use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxDtype, MlxRuntime};

pub(super) const EXPERT_COUNT: i32 = 4;
const HIDDEN_DIMENSION: i32 = 64;
const INTERMEDIATE_DIMENSION: i32 = 32;
const TOP_K: i32 = 2;
const QUANTIZATION_GROUP_SIZE: i32 = 32;
const QUANTIZATION_BIT_WIDTH: i32 = 4;

pub(super) struct RouteCase {
    pub(super) label: &'static str,
    pub(super) activations: MlxArray,
    pub(super) expert_indices: MlxArray,
    pub(super) are_indices_sorted: bool,
}

pub(super) fn route_cases(runtime: &MlxRuntime) -> [RouteCase; 2] {
    // The first shape matches one-token unsorted decode. The second models the
    // flattened assignment rows used after the existing >=64 assignment sort.
    let decode_hidden_states = bfloat16_values(
        runtime,
        HIDDEN_DIMENSION,
        &[1, 1, HIDDEN_DIMENSION],
        0.015625,
        -0.5,
    );
    let decode_activations = runtime
        .expand_dims(&decode_hidden_states, -2)
        .and_then(|expanded_states| runtime.expand_dims(&expanded_states, -3))
        .expect("decode hidden states should gain gathered projection axes");
    let decode_indices = runtime
        .array_from_u32(&[3, 1], &[1, 1, TOP_K])
        .expect("decode expert indices should allocate");
    let sorted_token_count = 40;
    let sorted_assignment_count = sorted_token_count * TOP_K;
    let sorted_hidden_states = bfloat16_values(
        runtime,
        sorted_token_count * HIDDEN_DIMENSION,
        &[1, sorted_token_count, HIDDEN_DIMENSION],
        0.00390625,
        -0.75,
    );
    let expanded_sorted_states = runtime
        .expand_dims(&sorted_hidden_states, -2)
        .and_then(|expanded_states| runtime.expand_dims(&expanded_states, -3))
        .expect("multi-token hidden states should gain gathered projection axes");
    let unsorted_indices = runtime
        .array_from_u32(
            &(0..sorted_assignment_count)
                .map(|assignment_index| {
                    u32::try_from((assignment_index * 3 + 1) % EXPERT_COUNT)
                        .expect("the sorted expert index should fit u32")
                })
                .collect::<Vec<_>>(),
            &[1, sorted_token_count, TOP_K],
        )
        .expect("multi-token expert indices should allocate");
    let (sorted_activations, sorted_indices, _inverse_order) =
        qwen3_5_moe_sort_expert_assignments(runtime, &expanded_sorted_states, &unsorted_indices)
            .expect("multi-token expert assignments should sort through production logic");
    [
        RouteCase {
            label: "one_token_unsorted",
            activations: decode_activations,
            expert_indices: decode_indices,
            are_indices_sorted: false,
        },
        RouteCase {
            label: "eighty_assignments_sorted",
            activations: sorted_activations,
            expert_indices: sorted_indices,
            are_indices_sorted: true,
        },
    ]
}

pub(super) enum NativeGateUpWeights<'weights> {
    Separate {
        gate: &'weights MlxArray,
        up: &'weights MlxArray,
    },
    Fused(&'weights MlxArray),
}

pub(super) fn native_expert_forward(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    route_case: &RouteCase,
    gate_up_weights: NativeGateUpWeights<'_>,
    down_weights: &MlxArray,
) -> MlxArray {
    let (gate_output, up_output) = match gate_up_weights {
        NativeGateUpWeights::Separate { gate, up } => (
            gather_native(runtime, route_case, gate),
            gather_native(runtime, route_case, up),
        ),
        NativeGateUpWeights::Fused(fused_weights) => {
            split_gate_up_output(runtime, &gather_native(runtime, route_case, fused_weights))
        }
    };
    let activated = runtime
        .apply_compiled_swiglu(compiled_swiglu, &gate_output, &up_output)
        .expect("native SwiGLU should build");
    gather_native_with_activations(runtime, route_case, &activated, down_weights)
}

fn gather_native(runtime: &MlxRuntime, route_case: &RouteCase, weights: &MlxArray) -> MlxArray {
    gather_native_with_activations(runtime, route_case, &route_case.activations, weights)
}

fn gather_native_with_activations(
    runtime: &MlxRuntime,
    route_case: &RouteCase,
    activations: &MlxArray,
    weights: &MlxArray,
) -> MlxArray {
    let transposed_weights = runtime
        .transpose_axes(weights, &[0, 2, 1])
        .expect("native expert weights should transpose");
    runtime
        .gather_dense_matmul(
            activations,
            &transposed_weights,
            None,
            Some(&route_case.expert_indices),
            route_case.are_indices_sorted,
        )
        .expect("native gathered projection should build")
}

pub(super) struct QuantizedWeights {
    pub(super) packed: MlxArray,
    pub(super) scales: MlxArray,
    pub(super) biases: MlxArray,
}

pub(super) enum QuantizedGateUpWeights<'weights> {
    Separate {
        gate: &'weights QuantizedWeights,
        up: &'weights QuantizedWeights,
    },
    Fused(&'weights QuantizedWeights),
}

pub(super) fn quantized_expert_forward(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    route_case: &RouteCase,
    gate_up_weights: QuantizedGateUpWeights<'_>,
    down_weights: &QuantizedWeights,
) -> MlxArray {
    let (gate_output, up_output) = match gate_up_weights {
        QuantizedGateUpWeights::Separate { gate, up } => (
            gather_quantized(runtime, route_case, &route_case.activations, gate),
            gather_quantized(runtime, route_case, &route_case.activations, up),
        ),
        QuantizedGateUpWeights::Fused(fused_weights) => split_gate_up_output(
            runtime,
            &gather_quantized(runtime, route_case, &route_case.activations, fused_weights),
        ),
    };
    let activated = runtime
        .apply_compiled_swiglu(compiled_swiglu, &gate_output, &up_output)
        .expect("quantized SwiGLU should build");
    gather_quantized(runtime, route_case, &activated, down_weights)
}

pub(super) fn gather_quantized(
    runtime: &MlxRuntime,
    route_case: &RouteCase,
    activations: &MlxArray,
    weights: &QuantizedWeights,
) -> MlxArray {
    runtime
        .gather_quantized_matmul_affine(
            activations,
            &weights.packed,
            &weights.scales,
            &weights.biases,
            None,
            Some(&route_case.expert_indices),
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BIT_WIDTH,
            route_case.are_indices_sorted,
        )
        .expect("affine gathered projection should build")
}

fn split_gate_up_output(runtime: &MlxRuntime, fused_output: &MlxArray) -> (MlxArray, MlxArray) {
    let output_shape = fused_output.shape();
    let last_dimension_index = output_shape.len() - 1;
    let projection_dimension = output_shape[last_dimension_index] / 2;
    let gate_starts = vec![0; output_shape.len()];
    let mut gate_stops = output_shape.clone();
    gate_stops[last_dimension_index] = projection_dimension;
    let mut up_starts = gate_starts.clone();
    up_starts[last_dimension_index] = projection_dimension;
    let slice_strides = vec![1; output_shape.len()];
    let gate_output = runtime
        .slice(fused_output, &gate_starts, &gate_stops, &slice_strides)
        .expect("the fused gate rows should slice from the first output half");
    let up_output = runtime
        .slice(fused_output, &up_starts, &output_shape, &slice_strides)
        .expect("the fused up rows should slice from the second output half");
    (gate_output, up_output)
}

pub(super) fn native_expert_weights(
    runtime: &MlxRuntime,
    multiplier: f32,
    offset: f32,
) -> MlxArray {
    bfloat16_values(
        runtime,
        EXPERT_COUNT * INTERMEDIATE_DIMENSION * HIDDEN_DIMENSION,
        &[EXPERT_COUNT, INTERMEDIATE_DIMENSION, HIDDEN_DIMENSION],
        multiplier,
        offset,
    )
}

pub(super) fn native_down_weights(runtime: &MlxRuntime) -> MlxArray {
    bfloat16_values(
        runtime,
        EXPERT_COUNT * HIDDEN_DIMENSION * INTERMEDIATE_DIMENSION,
        &[EXPERT_COUNT, HIDDEN_DIMENSION, INTERMEDIATE_DIMENSION],
        0.005859375,
        -0.125,
    )
}

pub(super) fn quantized_expert_weights(
    runtime: &MlxRuntime,
    multiplier: f32,
    offset: f32,
) -> QuantizedWeights {
    quantized_weights(
        runtime,
        INTERMEDIATE_DIMENSION,
        HIDDEN_DIMENSION,
        multiplier,
        offset,
    )
}

pub(super) fn quantized_down_weights(runtime: &MlxRuntime) -> QuantizedWeights {
    quantized_weights(
        runtime,
        HIDDEN_DIMENSION,
        INTERMEDIATE_DIMENSION,
        0.005859375,
        -0.125,
    )
}

fn quantized_weights(
    runtime: &MlxRuntime,
    output_dimension: i32,
    input_dimension: i32,
    multiplier: f32,
    offset: f32,
) -> QuantizedWeights {
    let dense_weights = float32_values(
        runtime,
        EXPERT_COUNT * output_dimension * input_dimension,
        &[EXPERT_COUNT, output_dimension, input_dimension],
        multiplier,
        offset,
    );
    let (packed, scales, biases) = runtime
        .quantize_affine(
            &dense_weights,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BIT_WIDTH,
        )
        .expect("deterministic expert weights should quantize through MLX");
    QuantizedWeights {
        packed,
        scales,
        biases,
    }
}

fn bfloat16_values(
    runtime: &MlxRuntime,
    element_count: i32,
    shape: &[i32],
    multiplier: f32,
    offset: f32,
) -> MlxArray {
    let float32_values = float32_values(runtime, element_count, shape, multiplier, offset);
    runtime
        .astype(&float32_values, MlxDtype::BFloat16)
        .expect("deterministic values should cast to BFloat16")
}

fn float32_values(
    runtime: &MlxRuntime,
    element_count: i32,
    shape: &[i32],
    multiplier: f32,
    offset: f32,
) -> MlxArray {
    let values = (0..element_count)
        .map(|element_index| ((element_index % 97) as f32) * multiplier + offset)
        .collect::<Vec<_>>();
    runtime
        .array_from_f32(&values, shape)
        .expect("deterministic float32 values should allocate")
}

pub(super) fn assert_exact_float_values(
    runtime: &MlxRuntime,
    expected: &MlxArray,
    actual: &MlxArray,
    storage_kind: &str,
    route_kind: &str,
) {
    // Cast both outputs only for host inspection. The model computations above
    // retain their original BFloat16 or affine-promoted arithmetic.
    let expected_float32 = runtime
        .astype(expected, MlxDtype::Float32)
        .expect("the separate output should cast for inspection");
    let actual_float32 = runtime
        .astype(actual, MlxDtype::Float32)
        .expect("the fused output should cast for inspection");
    let expected_values = expected_float32
        .to_vec_f32()
        .expect("the separate output should evaluate");
    let actual_values = actual_float32
        .to_vec_f32()
        .expect("the fused output should evaluate");
    assert_eq!(expected_values.len(), actual_values.len());
    let first_mismatch = expected_values
        .iter()
        .zip(&actual_values)
        .enumerate()
        .find(|(_, (expected_value, actual_value))| expected_value != actual_value);
    assert!(
        first_mismatch.is_none(),
        "fusing {storage_kind} gate/up rows changed {route_kind} output at {first_mismatch:?}"
    );
}
