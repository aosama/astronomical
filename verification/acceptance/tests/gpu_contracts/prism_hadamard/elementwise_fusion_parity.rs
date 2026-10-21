//! Bit-for-bit parity of the gated-delta elementwise fusions against the
//! composed launches they replace (issue #1366).
//!
//! The fusions only restructure host work — every primitive and its order is
//! the composed path's — so replaying both routes over adversarial inputs
//! must agree element for element in both activation dtypes the family can
//! serve. Any divergence here means the engine must keep the composed route,
//! because the upstream greedy parity journey replays token for token and a
//! single differing bit can steer it.
//!
//! A third fusion — the output RMS normalization chained into the precise
//! SwiGLU epilogue — passed this synthetic parity contract yet diverged the
//! real upstream trajectory, because MLX's compiler reshapes the reduction
//! schedule when the elementwise epilogue fuses at the model's real widths;
//! it was rejected by the trajectory bisect and stays composed. Synthetic
//! parity alone is therefore necessary but not sufficient: the trajectory
//! replay remains the deciding gate.

use astronomical_mlx_c_rust::{
    MlxArray, MlxCompiledDualHeadNorm, MlxCompiledElementwiseGraphs,
    MlxCompiledGatedDeltaRateActivation, MlxDtype,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const RMS_NORM_EPSILON: f32 = 1.0e-6;
const TOKEN_COUNT: i32 = 48;
const HEAD_COUNT: i32 = 8;
const HEAD_DIMENSION: i32 = 128;
/// Softplus saturation territory: decay inputs near and beyond where
/// `logaddexp(x, 0)` stops moving, plus deep negatives that underflow.
const RATE_EXTREMES: [f32; 6] = [-40.0, -8.0, -0.001, 0.001, 8.0, 40.0];

#[test]
#[serial]
#[ignore = "proves bit-for-bit parity of the gated-delta elementwise fusions on the graphics processor"]
fn should_replay_the_gated_delta_elementwise_fusions_bit_for_bit() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(2 * 1024 * 1024 * 1024, 1024 * 1024 * 1024)
            .expect("the parity memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let elementwise_graphs =
        MlxCompiledElementwiseGraphs::new().expect("the shared elementwise graphs should compile");
    let rate_activation = MlxCompiledGatedDeltaRateActivation::new()
        .expect("the rate-activation graph should compile");
    let dual_head_norm = MlxCompiledDualHeadNorm::new(RMS_NORM_EPSILON)
        .expect("the dual head-norm graph should compile");

    for activation_dtype in [MlxDtype::Float16, MlxDtype::BFloat16] {
        prove_rate_activation_parity(
            &runtime,
            &elementwise_graphs,
            &rate_activation,
            activation_dtype,
        );
        prove_dual_head_norm_parity(&runtime, &dual_head_norm, activation_dtype);
    }
}

fn prove_rate_activation_parity(
    runtime: &MlxRuntime,
    elementwise_graphs: &MlxCompiledElementwiseGraphs,
    rate_activation: &MlxCompiledGatedDeltaRateActivation,
    activation_dtype: MlxDtype,
) {
    let element_count = (TOKEN_COUNT * HEAD_COUNT) as usize;
    // Base sweep plus the softplus extremes interleaved so every head sees
    // saturation, underflow, and the near-zero transition band.
    let update_values: Vec<f32> = (0..element_count)
        .map(|element_index| {
            let extreme = RATE_EXTREMES[element_index % RATE_EXTREMES.len()];
            let base = (element_index % 251) as f32 * 0.004 - 0.5;
            if element_index % 2 == 0 {
                extreme
            } else {
                base
            }
        })
        .collect();
    let decay_input_values: Vec<f32> = (0..element_count)
        .map(|element_index| RATE_EXTREMES[(element_index + 2) % RATE_EXTREMES.len()])
        .collect();
    let decay_log_values: Vec<f32> = (0..HEAD_COUNT as usize)
        .map(|head_index| -0.05 - (head_index % 5) as f32 * 0.2)
        .collect();
    let decay_bias_values: Vec<f32> = (0..HEAD_COUNT as usize)
        .map(|head_index| 0.5 + (head_index % 3) as f32)
        .collect();

    let update_logits = cast_array(
        runtime,
        &update_values,
        &[1, TOKEN_COUNT, HEAD_COUNT],
        activation_dtype,
    );
    let decay_inputs = cast_array(
        runtime,
        &decay_input_values,
        &[1, TOKEN_COUNT, HEAD_COUNT],
        activation_dtype,
    );
    let decay_rate_logarithm =
        cast_array(runtime, &decay_log_values, &[HEAD_COUNT], MlxDtype::Float32);
    let decay_interval_bias =
        cast_array(runtime, &decay_bias_values, &[HEAD_COUNT], activation_dtype);

    let (fused_update_rates, fused_decays) = runtime
        .apply_compiled_gated_delta_rate_activation(
            rate_activation,
            &update_logits,
            &decay_inputs,
            &decay_rate_logarithm,
            &decay_interval_bias,
        )
        .expect("the fused rate activation should replay");
    let composed_update_rates = runtime
        .sigmoid(&update_logits)
        .expect("the composed sigmoid should evaluate");
    let composed_decays = runtime
        .apply_compiled_gated_delta_decay(
            elementwise_graphs,
            &decay_rate_logarithm,
            &decay_inputs,
            &decay_interval_bias,
        )
        .expect("the composed decay graph should replay");

    assert_bit_for_bit_parity(
        runtime,
        &fused_update_rates,
        &composed_update_rates,
        "rate activation: update rates",
    );
    assert_bit_for_bit_parity(
        runtime,
        &fused_decays,
        &composed_decays,
        "rate activation: decays",
    );
}

fn prove_dual_head_norm_parity(
    runtime: &MlxRuntime,
    dual_head_norm: &MlxCompiledDualHeadNorm,
    activation_dtype: MlxDtype,
) {
    let head_shape = [1, TOKEN_COUNT, HEAD_COUNT, HEAD_DIMENSION];
    let element_count = (TOKEN_COUNT * HEAD_COUNT * HEAD_DIMENSION) as usize;
    // Norms are sensitive to magnitude spread: mix a base sweep, exact zeros
    // (the epsilon path), and rows scaled to the dtype's upper range.
    let query_values: Vec<f32> = (0..element_count)
        .map(|element_index| match element_index % 7 {
            0 => 0.0,
            1 | 2 | 3 => (element_index % 521) as f32 * 0.002 - 0.5,
            _ => (element_index % 11) as f32 - 5.0,
        })
        .collect();
    let key_values: Vec<f32> = (0..element_count)
        .map(|element_index| match element_index % 5 {
            0 => 0.0,
            _ => ((element_index % 17) as f32 - 8.0) * 0.1,
        })
        .collect();
    let scale_values: Vec<f32> = (0..HEAD_DIMENSION as usize)
        .map(|channel_index| 0.75 + (channel_index % 4) as f32 * 0.1)
        .collect();

    let queries = cast_array(runtime, &query_values, &head_shape, activation_dtype);
    let keys = cast_array(runtime, &key_values, &head_shape, activation_dtype);
    let query_scale_weight =
        cast_array(runtime, &scale_values, &[HEAD_DIMENSION], activation_dtype);
    let key_scale_weight = cast_array(
        runtime,
        &scale_values.iter().rev().copied().collect::<Vec<_>>(),
        &[HEAD_DIMENSION],
        activation_dtype,
    );

    let (fused_queries, fused_keys) = runtime
        .apply_compiled_dual_head_norm(
            dual_head_norm,
            &queries,
            &keys,
            &query_scale_weight,
            &key_scale_weight,
        )
        .expect("the fused dual head norm should replay");
    let composed_queries = runtime
        .rms_norm(&queries, &query_scale_weight, RMS_NORM_EPSILON)
        .expect("the composed query norm should evaluate");
    let composed_keys = runtime
        .rms_norm(&keys, &key_scale_weight, RMS_NORM_EPSILON)
        .expect("the composed key norm should evaluate");

    assert_bit_for_bit_parity(
        runtime,
        &fused_queries,
        &composed_queries,
        "dual head norm: queries",
    );
    assert_bit_for_bit_parity(runtime, &fused_keys, &composed_keys, "dual head norm: keys");
}

fn cast_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32], dtype: MlxDtype) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(values, shape)
                .expect("the parity input should construct"),
            dtype,
        )
        .expect("the parity input should cast")
}

fn assert_bit_for_bit_parity(
    runtime: &MlxRuntime,
    fused: &MlxArray,
    composed: &MlxArray,
    label: &str,
) {
    let elementwise_equality = runtime
        .array_equal(fused, composed, true)
        .unwrap_or_else(|error| panic!("{label}: the equality probe failed: {error}"));
    let equality_flags = runtime
        .array_to_vec_f32(
            &runtime
                .astype(&elementwise_equality, MlxDtype::Float32)
                .unwrap_or_else(|error| panic!("{label}: the equality flags should cast: {error}")),
        )
        .unwrap_or_else(|error| panic!("{label}: the equality flags should read back: {error}"));
    assert!(
        equality_flags.iter().all(|flag| *flag != 0.0),
        "{label}: the fused replay diverged from the composed path"
    );
}
