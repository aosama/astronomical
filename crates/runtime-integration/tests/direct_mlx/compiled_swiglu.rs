use astronomical_runtime_integration::{MlxCompiledSwiGlu, MlxDtype, MlxRuntime};

use crate::common::runtime_test_support::{assert_bfloat16_arrays_match, runtime};

#[test]
fn should_reuse_one_shapeless_compiled_swiglu_across_sequence_lengths() {
    let runtime = runtime();
    let compiled_swiglu =
        MlxCompiledSwiGlu::new().expect("the shapeless SwiGLU graph should compile");

    assert_compiled_swiglu_matches_reference(
        &runtime,
        &compiled_swiglu,
        &[-1.0, 0.0, 1.0, 0.5],
        &[2.0, 3.0, 4.0, 5.0],
        &[1, 2, 2],
    );
    assert_compiled_swiglu_matches_reference(
        &runtime,
        &compiled_swiglu,
        &[2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25, 3.0, -3.0],
        &[1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 0.75, 1.25],
        &[1, 5, 2],
    );
}

fn assert_compiled_swiglu_matches_reference(
    runtime: &MlxRuntime,
    compiled_swiglu: &MlxCompiledSwiGlu,
    gate_values: &[f32],
    input_values: &[f32],
    shape: &[i32],
) {
    let gate = runtime
        .array_from_f32(gate_values, shape)
        .and_then(|float32_gate| runtime.astype(&float32_gate, MlxDtype::BFloat16))
        .expect("the gate should be valid");
    let input = runtime
        .array_from_f32(input_values, shape)
        .and_then(|float32_input| runtime.astype(&float32_input, MlxDtype::BFloat16))
        .expect("the input should be valid");
    let reference_output = runtime
        .silu(&gate)
        .and_then(|activated_gate| runtime.multiply(&activated_gate, &input))
        .expect("the uncompiled SwiGLU should build a graph");
    let compiled_output = runtime
        .apply_compiled_swiglu(compiled_swiglu, &gate, &input)
        .expect("the compiled SwiGLU should accept the runtime sequence length");

    assert_eq!(compiled_output.shape(), shape);
    assert_bfloat16_arrays_match(runtime, &compiled_output, &reference_output);
}
