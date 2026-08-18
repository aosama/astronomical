//! Direct GPU contracts for Astronomical's owned MLX custom Metal-kernel boundary.

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

#[test]
fn should_apply_a_custom_metal_kernel_with_a_small_uint32_output() {
    let runtime = runtime();
    let source_values = runtime
        .array_from_u32(&[u32::MAX, 0, 0], &[3])
        .expect("the uint32 source values should be valid");
    let copy_kernel = MlxMetalKernel::new(
        "astronomical_uint32_copy_kernel_test",
        &["source_values"],
        &["copied_values"],
        r#"
            for (uint element_index = 0; element_index < 3; ++element_index) {
                copied_values[element_index] = source_values[element_index];
            }
        "#,
    )
    .expect("the uint32 custom Metal kernel should be constructed");
    let mut copied_outputs = runtime
        .apply_metal_kernel(
            &copy_kernel,
            &[&source_values],
            &[MlxMetalKernelOutput::new(vec![3], MlxDtype::UInt32)],
            [1, 1, 1],
            [1, 1, 1],
            &[],
        )
        .expect("the uint32 custom Metal kernel should build a valid graph");
    let copied_values = copied_outputs
        .pop()
        .expect("the uint32 custom Metal kernel should return one output");

    assert_eq!(
        copied_values
            .to_vec_u32()
            .expect("the uint32 custom Metal output should evaluate"),
        vec![u32::MAX, 0, 0]
    );
}

#[test]
fn should_execute_same_name_different_source_kernels_in_one_evaluation_group() {
    let runtime = runtime();
    let source_values = runtime
        .array_from_f32(&[0.0, 1.0, 2.0, 3.0], &[4])
        .expect("the shared source values should be valid");
    // Real owners use unique names today, but MLX identifies compiled libraries
    // below this boundary. Keeping the collision in one evaluation group guards
    // against a dependency regression silently substituting another owner's code.
    let doubled_kernel = MlxMetalKernel::new(
        "astronomical_same_name_different_source_test",
        &["source_values"],
        &["transformed_values"],
        r#"
            uint element_index = thread_position_in_grid.x;
            transformed_values[element_index] = source_values[element_index] * 2.0f;
        "#,
    )
    .expect("the doubling custom Metal kernel should be constructed");
    let shifted_kernel = MlxMetalKernel::new(
        "astronomical_same_name_different_source_test",
        &["source_values"],
        &["transformed_values"],
        r#"
            uint element_index = thread_position_in_grid.x;
            transformed_values[element_index] = source_values[element_index] + 100.0f;
        "#,
    )
    .expect("the shifting custom Metal kernel should be constructed");

    let mut doubled_outputs = runtime
        .apply_metal_kernel(
            &doubled_kernel,
            &[&source_values],
            &[MlxMetalKernelOutput::new(vec![4], MlxDtype::Float32)],
            [4, 1, 1],
            [4, 1, 1],
            &[],
        )
        .expect("the doubling kernel should build a valid graph");
    let doubled_values = doubled_outputs
        .pop()
        .expect("the doubling kernel should return one output");
    let mut shifted_outputs = runtime
        .apply_metal_kernel(
            &shifted_kernel,
            &[&source_values],
            &[MlxMetalKernelOutput::new(vec![4], MlxDtype::Float32)],
            [4, 1, 1],
            [4, 1, 1],
            &[],
        )
        .expect("the shifting kernel should build a valid graph");
    let shifted_values = shifted_outputs
        .pop()
        .expect("the shifting kernel should return one output");

    runtime
        .evaluate_arrays(&[&doubled_values, &shifted_values])
        .expect("both same-name kernels should evaluate in one group");
    assert_f32_close(
        &doubled_values
            .to_vec_f32()
            .expect("the doubled values should be readable"),
        &[0.0, 2.0, 4.0, 6.0],
    );
    assert_f32_close(
        &shifted_values
            .to_vec_f32()
            .expect("the shifted values should be readable"),
        &[100.0, 101.0, 102.0, 103.0],
    );
}
