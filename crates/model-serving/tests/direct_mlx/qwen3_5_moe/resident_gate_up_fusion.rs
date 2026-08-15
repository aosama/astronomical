use std::time::Duration;

use astronomical_runtime_integration::{MlxArray, MlxCompiledSwiGlu, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use super::resident_gate_up_fusion_support::*;
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_preserve_resident_expert_outputs_after_gate_up_fusion() {
    timeout(TEST_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let runtime = MlxRuntime::initialize(
            MlxMemoryLimits::new(
                DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
                DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
            )
            .expect("the fusion parity memory limits should be valid"),
        )
        .expect("the direct MLX runtime should initialize for fusion parity");
        let compiled_swiglu =
            MlxCompiledSwiGlu::new().expect("the shared SwiGLU graph should compile");

        eprintln!("[resident-gate-up-fusion 1/4] status=progress storage=native_bfloat16");
        compare_native_bfloat16_routes(&runtime, &compiled_swiglu);
        eprintln!("[resident-gate-up-fusion 2/4] status=progress storage=affine_quantized");
        compare_affine_quantized_routes(&runtime, &compiled_swiglu);
        eprintln!("[resident-gate-up-fusion 3/4] status=progress phase=exact_parity_confirmed");
        eprintln!("[resident-gate-up-fusion 4/4] status=success");
    })
    .await
    .expect("resident gate/up fusion parity must finish within 115 seconds");
}

fn compare_native_bfloat16_routes(runtime: &MlxRuntime, compiled_swiglu: &MlxCompiledSwiGlu) {
    let gate_weights = native_expert_weights(runtime, 0.0078125, -0.25);
    let up_weights = native_expert_weights(runtime, 0.01171875, 0.125);
    let down_weights = native_down_weights(runtime);
    let concatenated_gate_up_weights = runtime
        .concatenate_axis(&[&gate_weights, &up_weights], 1)
        .expect("native gate and up rows should concatenate");
    let fused_gate_up_weights = runtime
        .build_contiguous_row_major_copy(&concatenated_gate_up_weights)
        .expect("native fused gate/up rows should become contiguous");
    runtime
        .evaluate_arrays(&[&fused_gate_up_weights])
        .expect("native fused gate/up rows should materialize before gathered execution");

    for route_case in route_cases(runtime) {
        let separate_output = native_expert_forward(
            runtime,
            compiled_swiglu,
            &route_case,
            NativeGateUpWeights::Separate {
                gate: &gate_weights,
                up: &up_weights,
            },
            &down_weights,
        );
        let fused_output = native_expert_forward(
            runtime,
            compiled_swiglu,
            &route_case,
            NativeGateUpWeights::Fused(&fused_gate_up_weights),
            &down_weights,
        );
        assert_exact_float_values(
            runtime,
            &separate_output,
            &fused_output,
            "native_bfloat16",
            route_case.label,
        );
    }
}

fn compare_affine_quantized_routes(runtime: &MlxRuntime, compiled_swiglu: &MlxCompiledSwiGlu) {
    let gate_weights = quantized_expert_weights(runtime, 0.0078125, -0.25);
    let up_weights = quantized_expert_weights(runtime, 0.01171875, 0.125);
    let down_weights = quantized_down_weights(runtime);
    let concatenated_gate_up_weights = QuantizedWeights {
        packed: runtime
            .concatenate_axis(&[&gate_weights.packed, &up_weights.packed], 1)
            .expect("packed gate and up rows should concatenate"),
        scales: runtime
            .concatenate_axis(&[&gate_weights.scales, &up_weights.scales], 1)
            .expect("gate and up scales should concatenate"),
        biases: runtime
            .concatenate_axis(&[&gate_weights.biases, &up_weights.biases], 1)
            .expect("gate and up biases should concatenate"),
    };
    let fused_gate_up_weights = QuantizedWeights {
        packed: runtime
            .build_contiguous_row_major_copy(&concatenated_gate_up_weights.packed)
            .expect("packed fused rows should become contiguous"),
        scales: runtime
            .build_contiguous_row_major_copy(&concatenated_gate_up_weights.scales)
            .expect("fused scales should become contiguous"),
        biases: runtime
            .build_contiguous_row_major_copy(&concatenated_gate_up_weights.biases)
            .expect("fused biases should become contiguous"),
    };
    runtime
        .evaluate_arrays(&[
            &fused_gate_up_weights.packed,
            &fused_gate_up_weights.scales,
            &fused_gate_up_weights.biases,
        ])
        .expect("affine fused gate/up rows should materialize before gathered execution");
    assert_quantized_fusion_layout(runtime, &gate_weights, &up_weights, &fused_gate_up_weights);

    for route_case in route_cases(runtime) {
        compare_quantized_gate_up_projection(
            runtime,
            &route_case,
            &gate_weights,
            &up_weights,
            &fused_gate_up_weights,
        );
        if route_case.are_indices_sorted {
            // The real-artifact journey below covers the complete sorted MoE
            // stack. This focused contract isolates the operation changed by
            // fusion and avoids coupling it to synthetic small-shape down-QMM.
            continue;
        }
        let separate_output = quantized_expert_forward(
            runtime,
            compiled_swiglu,
            &route_case,
            QuantizedGateUpWeights::Separate {
                gate: &gate_weights,
                up: &up_weights,
            },
            &down_weights,
        );
        let fused_output = quantized_expert_forward(
            runtime,
            compiled_swiglu,
            &route_case,
            QuantizedGateUpWeights::Fused(&fused_gate_up_weights),
            &down_weights,
        );
        assert_exact_float_values(
            runtime,
            &separate_output,
            &fused_output,
            "affine_quantized",
            route_case.label,
        );
    }
}

fn assert_quantized_fusion_layout(
    runtime: &MlxRuntime,
    gate_weights: &QuantizedWeights,
    up_weights: &QuantizedWeights,
    fused_gate_up_weights: &QuantizedWeights,
) {
    let gate_packed_words = runtime
        .copy_u32_values(&gate_weights.packed)
        .expect("gate packed words should copy");
    let up_packed_words = runtime
        .copy_u32_values(&up_weights.packed)
        .expect("up packed words should copy");
    let fused_packed_words = runtime
        .copy_u32_values(&fused_gate_up_weights.packed)
        .expect("fused packed words should copy");
    let words_per_projection_expert = gate_packed_words.len()
        / usize::try_from(EXPERT_COUNT).expect("expert count should fit usize");
    let mut expected_fused_packed_words = Vec::with_capacity(fused_packed_words.len());
    for expert_index in 0..usize::try_from(EXPERT_COUNT).expect("expert count should fit usize") {
        let expert_start = expert_index * words_per_projection_expert;
        let expert_end = expert_start + words_per_projection_expert;
        expected_fused_packed_words.extend_from_slice(&gate_packed_words[expert_start..expert_end]);
        expected_fused_packed_words.extend_from_slice(&up_packed_words[expert_start..expert_end]);
    }
    assert_eq!(
        expected_fused_packed_words, fused_packed_words,
        "packed fusion must preserve per-expert `[gate, up]` row order"
    );
    assert_fused_float_parameter_layout(
        &gate_weights.scales,
        &up_weights.scales,
        &fused_gate_up_weights.scales,
        "scales",
    );
    assert_fused_float_parameter_layout(
        &gate_weights.biases,
        &up_weights.biases,
        &fused_gate_up_weights.biases,
        "biases",
    );
}

fn assert_fused_float_parameter_layout(
    gate_parameter: &MlxArray,
    up_parameter: &MlxArray,
    fused_parameter: &MlxArray,
    parameter_name: &str,
) {
    let gate_values = gate_parameter
        .to_vec_f32()
        .expect("gate affine parameters should copy");
    let up_values = up_parameter
        .to_vec_f32()
        .expect("up affine parameters should copy");
    let fused_values = fused_parameter
        .to_vec_f32()
        .expect("fused affine parameters should copy");
    let values_per_projection_expert =
        gate_values.len() / usize::try_from(EXPERT_COUNT).expect("expert count should fit usize");
    let mut expected_fused_values = Vec::with_capacity(fused_values.len());
    for expert_index in 0..usize::try_from(EXPERT_COUNT).expect("expert count should fit usize") {
        let expert_start = expert_index * values_per_projection_expert;
        let expert_end = expert_start + values_per_projection_expert;
        expected_fused_values.extend_from_slice(&gate_values[expert_start..expert_end]);
        expected_fused_values.extend_from_slice(&up_values[expert_start..expert_end]);
    }
    assert_eq!(
        expected_fused_values, fused_values,
        "{parameter_name} fusion must preserve per-expert `[gate, up]` row order"
    );
}

fn compare_quantized_gate_up_projection(
    runtime: &MlxRuntime,
    route_case: &RouteCase,
    gate_weights: &QuantizedWeights,
    up_weights: &QuantizedWeights,
    fused_gate_up_weights: &QuantizedWeights,
) {
    let separate_gate =
        gather_quantized(runtime, route_case, &route_case.activations, gate_weights);
    let separate_up = gather_quantized(runtime, route_case, &route_case.activations, up_weights);
    let fused_gate_up = gather_quantized(
        runtime,
        route_case,
        &route_case.activations,
        fused_gate_up_weights,
    );
    let expected_gate_up = runtime
        .concatenate_axis(&[&separate_gate, &separate_up], -1)
        .expect("separate gate/up outputs should concatenate for parity inspection");
    assert_exact_float_values(
        runtime,
        &expected_gate_up,
        &fused_gate_up,
        "affine_quantized_gate_up",
        route_case.label,
    );
}
