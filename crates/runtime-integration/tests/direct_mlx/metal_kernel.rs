use astronomical_runtime_integration::{MlxDtype, MlxMetalKernel, MlxMetalKernelOutput};

use crate::common::runtime_test_support::{assert_f32_close, runtime};

#[test]
fn should_apply_a_custom_metal_kernel_to_an_mlx_array() {
    let runtime = runtime();
    let source_values = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("the source values should be valid");
    let copy_kernel = MlxMetalKernel::new(
        "astronomical_copy_kernel_test",
        &["source_values"],
        &["copied_values"],
        r#"
            uint element_index = thread_position_in_grid.x;
            copied_values[element_index] = source_values[element_index];
        "#,
    )
    .expect("the custom Metal kernel should be constructed");

    let mut copied_outputs = runtime
        .apply_metal_kernel(
            &copy_kernel,
            &[&source_values],
            &[MlxMetalKernelOutput::new(vec![2, 2], MlxDtype::Float32)],
            [4, 1, 1],
            [2, 1, 1],
            &[],
        )
        .expect("the custom Metal kernel should build a valid graph");

    assert_eq!(copied_outputs.len(), 1);
    let copied_values = copied_outputs
        .pop()
        .expect("the custom Metal kernel should return one output");
    assert_f32_close(
        &copied_values
            .to_vec_f32()
            .expect("the copied values should evaluate"),
        &[1.0, 2.0, 3.0, 4.0],
    );
}

#[test]
fn should_apply_a_custom_metal_kernel_with_a_scalar_integer_input() {
    let runtime = runtime();
    let source_values = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("the source values should be valid");
    let token_count = runtime
        .array_from_i32(&[3], &[])
        .expect("the scalar token count should be valid");
    let scalar_input_kernel = MlxMetalKernel::new(
        "astronomical_scalar_input_kernel_test",
        &["source_values", "token_count"],
        &["shifted_values"],
        r#"
            uint element_index = thread_position_in_grid.x;
            shifted_values[element_index] = source_values[element_index] + static_cast<float>(token_count);
        "#,
    )
    .expect("the custom Metal kernel should be constructed");

    let mut shifted_outputs = runtime
        .apply_metal_kernel(
            &scalar_input_kernel,
            &[&source_values, &token_count],
            &[MlxMetalKernelOutput::new(vec![2, 2], MlxDtype::Float32)],
            [4, 1, 1],
            [2, 1, 1],
            &[],
        )
        .expect("the custom Metal kernel should build a valid graph");

    assert_eq!(shifted_outputs.len(), 1);
    let shifted_values = shifted_outputs
        .pop()
        .expect("the custom Metal kernel should return one output");
    assert_f32_close(
        &shifted_values
            .to_vec_f32()
            .expect("the shifted values should evaluate"),
        &[4.0, 5.0, 6.0, 7.0],
    );
}
