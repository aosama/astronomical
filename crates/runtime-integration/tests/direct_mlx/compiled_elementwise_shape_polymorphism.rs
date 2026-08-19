//! Shape-polymorphism contracts for every retained elementwise MLX compilation.

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxDtype, MlxRuntime,
};

use crate::common::runtime_test_support::{
    assert_bfloat16_arrays_match, assert_f32_close, runtime, stable_softplus_reference,
};

const FEATURE_DIMENSION: i32 = 4;

#[test]
fn should_reuse_every_shapeless_elementwise_graph_across_sequence_lengths() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");

    for sequence_length in [2, 5] {
        assert_attention_output_gate_matches_reference(
            &runtime,
            &compiled_elementwise_graphs,
            sequence_length,
        );
        assert_sparse_shared_expert_combination_matches_reference(
            &runtime,
            &compiled_elementwise_graphs,
            sequence_length,
        );
        assert_precise_swiglu_matches_reference(
            &runtime,
            &compiled_elementwise_graphs,
            sequence_length,
        );
        assert_gated_delta_decay_matches_reference(
            &runtime,
            &compiled_elementwise_graphs,
            sequence_length,
        );
    }
}

fn assert_attention_output_gate_matches_reference(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    sequence_length: i32,
) {
    let activation_shape = activation_shape(sequence_length);
    let element_count = activation_element_count(sequence_length);
    let attention_output = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, 0.25, -1.0),
        &activation_shape,
    );
    let output_gate_logits = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, 0.1, -0.5),
        &activation_shape,
    );
    let reference_output = runtime
        .sigmoid(&output_gate_logits)
        .and_then(|output_gate_weights| runtime.multiply(&attention_output, &output_gate_weights))
        .expect("the uncompiled attention gate should build a graph");
    let compiled_output = runtime
        .apply_compiled_attention_output_gate(
            compiled_elementwise_graphs,
            &attention_output,
            &output_gate_logits,
        )
        .expect("the compiled attention gate should accept the runtime sequence length");

    assert_bfloat16_shape_and_values_match(
        runtime,
        &compiled_output,
        &reference_output,
        &activation_shape,
    );
}

fn assert_sparse_shared_expert_combination_matches_reference(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    sequence_length: i32,
) {
    let activation_shape = activation_shape(sequence_length);
    let gate_shape = [1, sequence_length, 1];
    let element_count = activation_element_count(sequence_length);
    let sparse_expert_output = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, 0.2, -0.75),
        &activation_shape,
    );
    let shared_expert_output = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, -0.15, 1.25),
        &activation_shape,
    );
    let shared_expert_gate_logits = bfloat16_array(
        runtime,
        &scaled_sequence(
            usize::try_from(sequence_length).expect("the test sequence length should fit usize"),
            0.4,
            -0.3,
        ),
        &gate_shape,
    );
    let reference_output = runtime
        .sigmoid(&shared_expert_gate_logits)
        .and_then(|shared_expert_gate_weights| {
            runtime.multiply(&shared_expert_output, &shared_expert_gate_weights)
        })
        .and_then(|gated_shared_expert_output| {
            runtime.add(&sparse_expert_output, &gated_shared_expert_output)
        })
        .expect("the uncompiled expert combination should build a graph");
    let compiled_output = runtime
        .apply_compiled_sparse_shared_expert_combination(
            compiled_elementwise_graphs,
            &sparse_expert_output,
            &shared_expert_output,
            &shared_expert_gate_logits,
        )
        .expect("the compiled expert combination should accept the runtime sequence length");

    assert_bfloat16_shape_and_values_match(
        runtime,
        &compiled_output,
        &reference_output,
        &activation_shape,
    );
}

fn assert_precise_swiglu_matches_reference(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    sequence_length: i32,
) {
    let activation_shape = [1, sequence_length, 2, 2];
    let element_count = activation_element_count(sequence_length);
    let up_states = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, 0.125, -0.25),
        &activation_shape,
    );
    let gate_states = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, -0.2, 0.8),
        &activation_shape,
    );
    let reference_output = runtime
        .astype(&gate_states, MlxDtype::Float32)
        .and_then(|float32_gate_states| runtime.silu(&float32_gate_states))
        .and_then(|activated_gate| {
            runtime
                .astype(&up_states, MlxDtype::Float32)
                .and_then(|float32_up_states| runtime.multiply(&activated_gate, &float32_up_states))
        })
        .and_then(|float32_output| runtime.astype(&float32_output, MlxDtype::BFloat16))
        .expect("the uncompiled precise SwiGLU should build a graph");
    let compiled_output = runtime
        .apply_compiled_precise_swiglu(compiled_elementwise_graphs, &up_states, &gate_states)
        .expect("the compiled precise SwiGLU should accept the runtime sequence length");

    assert_bfloat16_shape_and_values_match(
        runtime,
        &compiled_output,
        &reference_output,
        &activation_shape,
    );
}

fn assert_gated_delta_decay_matches_reference(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    sequence_length: i32,
) {
    let activation_shape = activation_shape(sequence_length);
    let element_count = activation_element_count(sequence_length);
    let decay_rate_logarithm = bfloat16_array(runtime, &[-2.0, -1.0, -0.5, -0.25], &[4]);
    let decay_interval_inputs = bfloat16_array(
        runtime,
        &scaled_sequence(element_count, 0.1, -0.4),
        &activation_shape,
    );
    let decay_interval_bias = bfloat16_array(runtime, &[-0.5, 0.0, 0.25, 0.75], &[4]);
    let reference_output = runtime
        .add(&decay_interval_inputs, &decay_interval_bias)
        .and_then(|biased_decay_intervals| {
            stable_softplus_reference(runtime, &biased_decay_intervals)
        })
        .and_then(|decay_intervals| {
            runtime
                .astype(&decay_rate_logarithm, MlxDtype::Float32)
                .and_then(|float32_decay_logs| runtime.exp(&float32_decay_logs))
                .map(|decay_rates| (decay_rates, decay_intervals))
        })
        .and_then(|(decay_rates, decay_intervals)| runtime.multiply(&decay_rates, &decay_intervals))
        .and_then(|decay_products| runtime.negative(&decay_products))
        .and_then(|negative_decay_products| runtime.exp(&negative_decay_products))
        .expect("the uncompiled gated-delta decay should build a graph");
    let compiled_output = runtime
        .apply_compiled_gated_delta_decay(
            compiled_elementwise_graphs,
            &decay_rate_logarithm,
            &decay_interval_inputs,
            &decay_interval_bias,
        )
        .expect("the compiled gated-delta decay should accept the runtime sequence length");

    assert_float32_shape_and_values_match(&compiled_output, &reference_output, &activation_shape);
}

fn activation_shape(sequence_length: i32) -> [i32; 3] {
    [1, sequence_length, FEATURE_DIMENSION]
}

fn activation_element_count(sequence_length: i32) -> usize {
    usize::try_from(sequence_length * FEATURE_DIMENSION)
        .expect("the test activation element count should fit usize")
}

fn scaled_sequence(element_count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..element_count)
        .map(|element_index| element_index as f32 * scale + offset)
        .collect()
}

fn bfloat16_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    runtime
        .array_from_f32(values, shape)
        .and_then(|float32_array| runtime.astype(&float32_array, MlxDtype::BFloat16))
        .expect("the bfloat16 test array should be valid")
}

fn assert_bfloat16_shape_and_values_match(
    runtime: &MlxRuntime,
    compiled_output: &MlxArray,
    reference_output: &MlxArray,
    expected_shape: &[i32],
) {
    assert_eq!(compiled_output.shape(), expected_shape);
    assert_bfloat16_arrays_match(runtime, compiled_output, reference_output);
}

fn assert_float32_shape_and_values_match(
    compiled_output: &MlxArray,
    reference_output: &MlxArray,
    expected_shape: &[i32],
) {
    assert_eq!(compiled_output.shape(), expected_shape);
    assert_f32_close(
        &compiled_output
            .to_vec_f32()
            .expect("the compiled output should evaluate as float32"),
        &reference_output
            .to_vec_f32()
            .expect("the reference output should evaluate as float32"),
    );
}
