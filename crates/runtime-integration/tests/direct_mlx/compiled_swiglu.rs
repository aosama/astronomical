use astronomical_runtime_integration::MlxCompiledSwiGlu;

use crate::common::runtime_test_support::{assert_f32_close, runtime};

#[test]
fn should_reuse_one_shapeless_compiled_swiglu_across_input_shapes() {
    let runtime = runtime();
    let compiled_swiglu =
        MlxCompiledSwiGlu::new().expect("the shapeless SwiGLU graph should compile");
    let first_gate = runtime
        .array_from_f32(&[-1.0, 0.0, 1.0], &[3])
        .expect("the first gate should be valid");
    let first_input = runtime
        .array_from_f32(&[2.0, 3.0, 4.0], &[3])
        .expect("the first input should be valid");

    let first_output = runtime
        .apply_compiled_swiglu(&compiled_swiglu, &first_gate, &first_input)
        .expect("the first compiled SwiGLU application should build a graph");

    assert_f32_close(
        &first_output
            .to_vec_f32()
            .expect("the first output should evaluate as float32"),
        &[-0.537_882_86, 0.0, 2.924_234_4],
    );

    let second_gate = runtime
        .array_from_f32(&[2.0, -2.0, 0.5, -0.5], &[2, 2])
        .expect("the second gate should be valid");
    let second_input = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("the second input should be valid");

    let second_output = runtime
        .apply_compiled_swiglu(&compiled_swiglu, &second_gate, &second_input)
        .expect("the retained compiled SwiGLU should accept a second shape");

    assert_f32_close(
        &second_output
            .to_vec_f32()
            .expect("the second output should evaluate as float32"),
        &[1.761_594, -0.476_811_68, 0.933_689, -0.755_081_36],
    );
}
