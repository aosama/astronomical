use std::time::{Duration, Instant};

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxDtype, MlxRuntime,
};

use crate::common::runtime_test_support::{
    assert_bfloat16_arrays_match, assert_f32_close, runtime, stable_softplus_reference,
};

#[test]
fn should_gate_full_attention_output_with_one_shapeless_compiled_graph() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let attention_output = runtime
        .array_from_f32(&[2.0, 4.0, 6.0, 8.0], &[2, 2])
        .expect("the attention output should be valid");
    let output_gate_logits = runtime
        .array_from_f32(&[0.0, 0.0, 0.0, 0.0], &[2, 2])
        .expect("the output gate logits should be valid");

    let gated_attention_output = runtime
        .apply_compiled_attention_output_gate(
            &compiled_elementwise_graphs,
            &attention_output,
            &output_gate_logits,
        )
        .expect("the compiled output gate should build a graph");

    assert_f32_close(
        &gated_attention_output
            .to_vec_f32()
            .expect("the gated output should evaluate as float32"),
        &[1.0, 2.0, 3.0, 4.0],
    );
}

#[test]
fn should_combine_sparse_and_gated_shared_experts_with_broadcast_gate_logits() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let sparse_expert_output = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("the sparse expert output should be valid");
    let shared_expert_output = runtime
        .array_from_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2])
        .expect("the shared expert output should be valid");
    let shared_expert_gate_logits = runtime
        .array_from_f32(&[0.0, 3.0_f32.ln()], &[2, 1])
        .expect("the shared expert gate logits should be valid");

    let combined_expert_output = runtime
        .apply_compiled_sparse_shared_expert_combination(
            &compiled_elementwise_graphs,
            &sparse_expert_output,
            &shared_expert_output,
            &shared_expert_gate_logits,
        )
        .expect("the compiled expert combination should build a graph");

    assert_f32_close(
        &combined_expert_output
            .to_vec_f32()
            .expect("the combined output should evaluate as float32"),
        &[6.0, 12.0, 25.5, 34.0],
    );
}

#[test]
fn should_match_precise_bfloat16_swiglu_with_one_compiled_graph() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let up_states = runtime
        .array_from_f32(&[2.0, -3.0, 4.0, -5.0], &[2, 2])
        .and_then(|float32_states| runtime.astype(&float32_states, MlxDtype::BFloat16))
        .expect("the up states should be valid bfloat16 values");
    let gate_states = runtime
        .array_from_f32(&[-1.0, 0.5, 1.0, -0.5], &[2, 2])
        .and_then(|float32_states| runtime.astype(&float32_states, MlxDtype::BFloat16))
        .expect("the gate states should be valid bfloat16 values");
    let reference_output = runtime
        .astype(&gate_states, MlxDtype::Float32)
        .and_then(|float32_gate| runtime.silu(&float32_gate))
        .and_then(|activated_gate| {
            runtime
                .astype(&up_states, MlxDtype::Float32)
                .and_then(|float32_up| runtime.multiply(&activated_gate, &float32_up))
        })
        .and_then(|activated_states| runtime.astype(&activated_states, MlxDtype::BFloat16))
        .expect("the reference precise SwiGLU should build a graph");

    let compiled_output = runtime
        .apply_compiled_precise_swiglu(&compiled_elementwise_graphs, &up_states, &gate_states)
        .expect("the compiled precise SwiGLU should build a graph");

    assert_eq!(compiled_output.dtype(), MlxDtype::BFloat16);
    let float32_compiled_output = runtime
        .astype(&compiled_output, MlxDtype::Float32)
        .expect("the compiled output should cast to float32 for comparison");
    let float32_reference_output = runtime
        .astype(&reference_output, MlxDtype::Float32)
        .expect("the reference output should cast to float32 for comparison");
    assert_f32_close(
        &float32_compiled_output
            .to_vec_f32()
            .expect("the compiled output should evaluate as float32 values"),
        &float32_reference_output
            .to_vec_f32()
            .expect("the reference output should evaluate as float32 values"),
    );
}

#[test]
fn should_match_precise_bfloat16_swiglu_with_one_token_compiled_graph() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let up_states = bfloat16_array(
        &runtime,
        &[2.0, -3.0, 4.0, -5.0, 0.25, -0.75, 1.5, -2.5],
        &[1, 1, 8],
    );
    let gate_states = bfloat16_array(
        &runtime,
        &[-1.0, 0.5, 1.0, -0.5, -0.25, 0.75, -1.5, 2.5],
        &[1, 1, 8],
    );
    let reference_output = runtime
        .astype(&gate_states, MlxDtype::Float32)
        .and_then(|float32_gate| runtime.silu(&float32_gate))
        .and_then(|activated_gate| {
            runtime
                .astype(&up_states, MlxDtype::Float32)
                .and_then(|float32_up| runtime.multiply(&activated_gate, &float32_up))
        })
        .and_then(|activated_states| runtime.astype(&activated_states, MlxDtype::BFloat16))
        .expect("the one-token reference precise SwiGLU should build a graph");

    let compiled_output = runtime
        .apply_compiled_precise_swiglu(&compiled_elementwise_graphs, &up_states, &gate_states)
        .expect("the one-token compiled precise SwiGLU should build a graph");

    assert_bfloat16_arrays_match(&runtime, &compiled_output, &reference_output);
}

#[test]
#[ignore = "manual one-token GPU benchmark; run explicitly with its 120-second command timeout"]
fn should_measure_warmed_one_token_precise_swiglu_paths() {
    const LINEAR_VALUE_DIMENSION: i32 = 4_096;
    const WARMUP_ITERATION_COUNT: usize = 20;
    const MEASUREMENT_ITERATION_COUNT: usize = 200;
    const MAXIMUM_MEASUREMENT_DURATION: Duration = Duration::from_secs(110);

    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let up_states = bfloat16_array(
        &runtime,
        &vec![0.25; LINEAR_VALUE_DIMENSION as usize],
        &[1, 1, LINEAR_VALUE_DIMENSION],
    );
    let gate_states = bfloat16_array(
        &runtime,
        &vec![-0.5; LINEAR_VALUE_DIMENSION as usize],
        &[1, 1, LINEAR_VALUE_DIMENSION],
    );

    let uncompiled_elapsed = measure_warmed_one_token_precise_swiglu_path(
        WARMUP_ITERATION_COUNT,
        MEASUREMENT_ITERATION_COUNT,
        MAXIMUM_MEASUREMENT_DURATION,
        || apply_uncompiled_precise_swiglu(&runtime, &up_states, &gate_states),
    );
    let compiled_elapsed = measure_warmed_one_token_precise_swiglu_path(
        WARMUP_ITERATION_COUNT,
        MEASUREMENT_ITERATION_COUNT,
        MAXIMUM_MEASUREMENT_DURATION,
        || {
            runtime.apply_compiled_precise_swiglu(
                &compiled_elementwise_graphs,
                &up_states,
                &gate_states,
            )
        },
    );

    eprintln!(
        "[one-token-precise-swiglu] status=success shape=[1,1,{LINEAR_VALUE_DIMENSION}] iterations={MEASUREMENT_ITERATION_COUNT} uncompiled_total_millis={} compiled_total_millis={} uncompiled_per_iteration_micros={:.1} compiled_per_iteration_micros={:.1}",
        uncompiled_elapsed.as_millis(),
        compiled_elapsed.as_millis(),
        uncompiled_elapsed.as_secs_f64() * 1_000_000.0 / MEASUREMENT_ITERATION_COUNT as f64,
        compiled_elapsed.as_secs_f64() * 1_000_000.0 / MEASUREMENT_ITERATION_COUNT as f64,
    );
}

#[test]
fn should_match_gated_delta_decay_arithmetic_with_one_compiled_graph() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let decay_rate_logarithm = runtime
        .array_from_f32(&[0.0, 2.0_f32.ln()], &[2])
        .expect("the decay rate logarithm should be valid");
    let decay_interval_inputs = runtime
        .array_from_f32(&[-1.0, 0.0, 1.0, 2.0], &[2, 2])
        .expect("the decay interval inputs should be valid");
    let decay_interval_bias = runtime
        .array_from_f32(&[0.5, -0.5], &[2])
        .expect("the decay interval bias should be valid");
    let reference_decays = runtime
        .add(&decay_interval_inputs, &decay_interval_bias)
        .and_then(|biased_intervals| stable_softplus_reference(&runtime, &biased_intervals))
        .and_then(|decay_intervals| {
            runtime
                .astype(&decay_rate_logarithm, MlxDtype::Float32)
                .and_then(|float32_decay_logs| runtime.exp(&float32_decay_logs))
                .and_then(|decay_rates| runtime.multiply(&decay_rates, &decay_intervals))
        })
        .and_then(|decay_products| runtime.negative(&decay_products))
        .and_then(|negative_decay_products| runtime.exp(&negative_decay_products))
        .expect("the reference decay arithmetic should build a graph");

    let compiled_decays = runtime
        .apply_compiled_gated_delta_decay(
            &compiled_elementwise_graphs,
            &decay_rate_logarithm,
            &decay_interval_inputs,
            &decay_interval_bias,
        )
        .expect("the compiled decay arithmetic should build a graph");

    assert_f32_close(
        &compiled_decays
            .to_vec_f32()
            .expect("the compiled decays should evaluate as float32"),
        &reference_decays
            .to_vec_f32()
            .expect("the reference decays should evaluate as float32"),
    );
}

#[test]
fn should_match_uncompiled_bfloat16_attention_gating_across_nontrivial_logits() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let attention_output = bfloat16_array(
        &runtime,
        &[-12.0, -1.5, -0.125, 0.0, 0.125, 1.5, 12.0, 31.0],
        &[2, 4],
    );
    let output_gate_logits = bfloat16_array(
        &runtime,
        &[-9.0, -2.0, -0.25, 0.0, 0.25, 2.0, 9.0, -4.0],
        &[2, 4],
    );
    let reference_output = runtime
        .sigmoid(&output_gate_logits)
        .and_then(|output_gate_weights| runtime.multiply(&attention_output, &output_gate_weights))
        .expect("the uncompiled attention gate should build a graph");
    let compiled_output = runtime
        .apply_compiled_attention_output_gate(
            &compiled_elementwise_graphs,
            &attention_output,
            &output_gate_logits,
        )
        .expect("the compiled attention gate should build a graph");

    assert_bfloat16_arrays_match(&runtime, &compiled_output, &reference_output);
}

#[test]
fn should_match_uncompiled_bfloat16_sparse_shared_expert_combination() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let sparse_expert_output = bfloat16_array(
        &runtime,
        &[-8.0, -1.0, 0.125, 3.0, 7.5, 11.0, -5.0, 0.0],
        &[2, 4],
    );
    let shared_expert_output = bfloat16_array(
        &runtime,
        &[4.0, -2.0, 1.0, 9.0, -3.0, 6.0, 12.0, -7.0],
        &[2, 4],
    );
    let shared_expert_gate_logits = bfloat16_array(&runtime, &[-2.25, 1.75], &[2, 1]);
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
            &compiled_elementwise_graphs,
            &sparse_expert_output,
            &shared_expert_output,
            &shared_expert_gate_logits,
        )
        .expect("the compiled expert combination should build a graph");

    assert_bfloat16_arrays_match(&runtime, &compiled_output, &reference_output);
}

#[test]
fn should_match_uncompiled_bfloat16_gated_delta_decay_across_wide_intervals() {
    let runtime = runtime();
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the shapeless elementwise graphs should compile");
    let decay_rate_logarithm = bfloat16_array(&runtime, &[-2.0, 0.0, 2.0], &[3]);
    let decay_interval_inputs = bfloat16_array(
        &runtime,
        &[-30.0, -8.0, -2.0, -0.25, 0.0, 0.25, 2.0, 8.0, 30.0],
        &[3, 3],
    );
    let decay_interval_bias = bfloat16_array(&runtime, &[-0.5, 0.25, 1.0], &[3]);
    let reference_decays = runtime
        .add(&decay_interval_inputs, &decay_interval_bias)
        .and_then(|biased_decay_intervals| {
            stable_softplus_reference(&runtime, &biased_decay_intervals)
        })
        .and_then(|decay_intervals| {
            runtime
                .astype(&decay_rate_logarithm, MlxDtype::Float32)
                .and_then(|float32_decay_logs| runtime.exp(&float32_decay_logs))
                .and_then(|decay_rates| runtime.multiply(&decay_rates, &decay_intervals))
        })
        .and_then(|decay_products| runtime.negative(&decay_products))
        .and_then(|negative_decay_products| runtime.exp(&negative_decay_products))
        .expect("the uncompiled gated-delta decay should build a graph");
    let compiled_decays = runtime
        .apply_compiled_gated_delta_decay(
            &compiled_elementwise_graphs,
            &decay_rate_logarithm,
            &decay_interval_inputs,
            &decay_interval_bias,
        )
        .expect("the compiled gated-delta decay should build a graph");

    // The decay formula is computed in float32 by design (matching the reference
    // `compute_g` which casts A_log to float32), so both paths return float32.
    assert_eq!(compiled_decays.dtype(), MlxDtype::Float32);
    assert_eq!(reference_decays.dtype(), MlxDtype::Float32);
    assert_f32_close(
        &compiled_decays
            .to_vec_f32()
            .expect("the compiled decays should evaluate"),
        &reference_decays
            .to_vec_f32()
            .expect("the reference decays should evaluate"),
    );
}

fn bfloat16_array(
    runtime: &astronomical_runtime_integration::MlxRuntime,
    float32_values: &[f32],
    shape: &[i32],
) -> astronomical_runtime_integration::MlxArray {
    runtime
        .array_from_f32(float32_values, shape)
        .and_then(|float32_array| runtime.astype(&float32_array, MlxDtype::BFloat16))
        .expect("the bfloat16 test array should be valid")
}

fn apply_uncompiled_precise_swiglu(
    runtime: &MlxRuntime,
    up_states: &MlxArray,
    gate_states: &MlxArray,
) -> Result<MlxArray, astronomical_runtime_integration::MlxRuntimeError> {
    let float32_gate_states = runtime.astype(gate_states, MlxDtype::Float32)?;
    let activated_gate_states = runtime.silu(&float32_gate_states)?;
    let float32_up_states = runtime.astype(up_states, MlxDtype::Float32)?;
    let activated_states = runtime.multiply(&activated_gate_states, &float32_up_states)?;
    runtime.astype(&activated_states, MlxDtype::BFloat16)
}

fn measure_warmed_one_token_precise_swiglu_path(
    warmup_iteration_count: usize,
    measurement_iteration_count: usize,
    maximum_measurement_duration: Duration,
    apply_precise_swiglu: impl Fn()
        -> Result<MlxArray, astronomical_runtime_integration::MlxRuntimeError>,
) -> Duration {
    for _warmup_iteration_index in 0..warmup_iteration_count {
        let output_states = apply_precise_swiglu().expect("the warmup SwiGLU graph should build");
        output_states
            .evaluate()
            .expect("the warmup SwiGLU output should evaluate");
    }

    let measurement_started_at = Instant::now();
    for _measurement_iteration_index in 0..measurement_iteration_count {
        let output_states = apply_precise_swiglu().expect("the measured SwiGLU graph should build");
        output_states
            .evaluate()
            .expect("the measured SwiGLU output should evaluate");
        assert!(
            measurement_started_at.elapsed() <= maximum_measurement_duration,
            "the one-token SwiGLU measurement exceeded the {}-second timeout",
            maximum_measurement_duration.as_secs()
        );
    }
    measurement_started_at.elapsed()
}
