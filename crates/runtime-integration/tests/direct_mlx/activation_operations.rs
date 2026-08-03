use astronomical_runtime_integration::MlxDtype;

use crate::common::runtime_test_support::{
    assert_bfloat16_arrays_match, assert_f32_close, runtime, stable_softplus_reference,
};

#[test]
fn should_apply_softplus_and_silu_on_the_runtime_stream() {
    let runtime = runtime();
    let activation_inputs = runtime
        .array_from_f32(&[-1.0, 0.0, 1.0], &[3])
        .expect("activation inputs should be valid");

    let softplus_values = runtime
        .softplus(&activation_inputs)
        .expect("softplus should build a valid graph");
    let silu_values = runtime
        .silu(&activation_inputs)
        .expect("silu should build a valid graph");

    assert_f32_close(
        &softplus_values
            .to_vec_f32()
            .expect("softplus should evaluate as float32"),
        &[0.313_261_7, std::f32::consts::LN_2, 1.313_261_6],
    );
    assert_f32_close(
        &silu_values
            .to_vec_f32()
            .expect("silu should evaluate as float32"),
        &[-0.268_941_43, 0.0, 0.731_058_6],
    );
}

#[test]
fn should_keep_softplus_finite_for_large_positive_inputs_without_overflow() {
    let runtime = runtime();
    let activation_inputs = runtime
        .array_from_f32(&[-100.0, -10.0, 0.0, 10.0, 100.0], &[5])
        .expect("activation inputs should be valid");

    let softplus_values = runtime
        .softplus(&activation_inputs)
        .expect("softplus should build a valid graph");

    let softplus_outputs = softplus_values
        .to_vec_f32()
        .expect("softplus should evaluate as float32");
    assert!(
        softplus_outputs.iter().all(|value| f32::is_finite(*value)),
        "softplus must stay finite for large inputs; got {softplus_outputs:?}"
    );
    // softplus(100) ≈ 100, softplus(10) ≈ 10.000045, softplus(0) = ln(2),
    // softplus(-10) ≈ 4.5e-5, softplus(-100) ≈ 0.
    assert_f32_close(
        &softplus_outputs,
        &[0.0, 0.000045_402, std::f32::consts::LN_2, 10.000045, 100.0],
    );
}

#[test]
fn should_apply_the_pytorch_tanh_gelu_approximation_on_the_runtime_stream() {
    let runtime = runtime();
    let activation_inputs = runtime
        .array_from_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5])
        .expect("activation inputs should be valid");

    let gelu_activation_values = runtime
        .gelu_tanh(&activation_inputs)
        .expect("tanh GELU should build a valid graph");

    assert_f32_close(
        &gelu_activation_values
            .to_vec_f32()
            .expect("tanh GELU should evaluate as float32"),
        &[-0.045_402_29, -0.158_808, 0.0, 0.841_192, 1.954_597_7],
    );

    let bfloat16_activation_inputs = runtime
        .astype(&activation_inputs, MlxDtype::BFloat16)
        .expect("activation inputs should cast to bfloat16");
    let bfloat16_gelu_activation_values = runtime
        .gelu_tanh(&bfloat16_activation_inputs)
        .expect("bfloat16 tanh GELU should build a valid graph");
    assert_eq!(
        bfloat16_gelu_activation_values.dtype(),
        MlxDtype::BFloat16,
        "tanh GELU must preserve the activation dtype"
    );
}

#[test]
fn should_apply_exact_gelu_on_the_runtime_stream() {
    let runtime = runtime();
    let activation_inputs = runtime
        .array_from_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5])
        .expect("activation inputs should be valid");

    let gelu_activation_values = runtime
        .gelu(&activation_inputs)
        .expect("exact GELU should build a valid graph");

    assert_f32_close(
        &gelu_activation_values
            .to_vec_f32()
            .expect("exact GELU should evaluate as float32"),
        &[-0.045_500_264, -0.158_655_26, 0.0, 0.841_344_7, 1.954_499_7],
    );
}

#[test]
fn should_apply_trigonometry_and_power_on_the_runtime_stream() {
    let runtime = runtime();
    let angles = runtime
        .array_from_f32(&[0.0, std::f32::consts::PI], &[2])
        .expect("the angle inputs should be valid");
    let power_base = runtime
        .array_from_f32(&[3.0], &[1])
        .expect("the power base should be valid");
    let power_exponent = runtime
        .array_from_f32(&[2.0], &[1])
        .expect("the power exponent should be valid");

    let cosine_values = runtime
        .cos(&angles)
        .expect("cosine should build a valid graph");
    let sine_values = runtime
        .sin(&angles)
        .expect("sine should build a valid graph");
    let power_values = runtime
        .power(&power_base, &power_exponent)
        .expect("power should build a valid graph");

    assert_f32_close(
        &cosine_values.to_vec_f32().expect("cosine should evaluate"),
        &[1.0, -1.0],
    );
    assert_f32_close(
        &sine_values.to_vec_f32().expect("sine should evaluate"),
        &[0.0, 0.0],
    );
    assert_f32_close(
        &power_values.to_vec_f32().expect("power should evaluate"),
        &[9.0],
    );
}

#[test]
fn should_apply_logaddexp_with_a_broadcast_scalar() {
    let runtime = runtime();
    let activation_inputs = runtime
        .array_from_f32(&[0.0, 10_000.0], &[2])
        .expect("activation inputs should be valid");
    let zero_scalar = runtime
        .zeros(&[], MlxDtype::Float32)
        .expect("the broadcast zero scalar should be valid");
    let logaddexp_values = runtime
        .logaddexp(&activation_inputs, &zero_scalar)
        .expect("logaddexp(x, 0) should build a valid graph");

    assert_f32_close(
        &logaddexp_values
            .to_vec_f32()
            .expect("logaddexp should evaluate as float32"),
        &[std::f32::consts::LN_2, 10_000.0],
    );
}

#[test]
fn should_match_stable_softplus_reference_for_bfloat16_inputs() {
    let runtime = runtime();
    let float32_inputs = runtime
        .array_from_f32(&[-30.0, -8.0, -2.0, -0.25, 0.0, 0.25, 2.0, 8.0, 30.0], &[9])
        .expect("float32 activation inputs should be valid");
    let bfloat16_inputs = runtime
        .astype(&float32_inputs, MlxDtype::BFloat16)
        .expect("activation inputs should cast to bfloat16");

    let softplus_values = runtime
        .softplus(&bfloat16_inputs)
        .expect("softplus should build a valid graph for bfloat16");
    let reference_softplus_values = stable_softplus_reference(&runtime, &bfloat16_inputs)
        .expect("the independent stable softplus reference should build a valid graph");

    assert_bfloat16_arrays_match(&runtime, &softplus_values, &reference_softplus_values);
}
