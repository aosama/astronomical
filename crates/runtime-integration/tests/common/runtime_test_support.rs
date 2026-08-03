#![allow(dead_code)]

use astronomical_runtime_integration::{
    MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime, MlxRuntimeError,
};

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;

pub fn runtime() -> MlxRuntime {
    let memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize")
}

pub fn assert_f32_close(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        assert!(
            (*actual_value - *expected_value).abs() <= 1e-6,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}

pub fn assert_bfloat16_arrays_match(
    runtime: &MlxRuntime,
    actual_array: &astronomical_runtime_integration::MlxArray,
    expected_array: &astronomical_runtime_integration::MlxArray,
) {
    assert_eq!(actual_array.dtype(), MlxDtype::BFloat16);
    assert_eq!(expected_array.dtype(), MlxDtype::BFloat16);
    let float32_actual_array = runtime
        .astype(actual_array, MlxDtype::Float32)
        .expect("the actual bfloat16 array should cast to float32");
    let float32_expected_array = runtime
        .astype(expected_array, MlxDtype::Float32)
        .expect("the expected bfloat16 array should cast to float32");
    assert_f32_close(
        &float32_actual_array
            .to_vec_f32()
            .expect("the actual array should evaluate"),
        &float32_expected_array
            .to_vec_f32()
            .expect("the expected array should evaluate"),
    );
}

/// Independent stable-softplus oracle retained to detect accidental changes in
/// the production `logaddexp(input, 0)` implementation.
pub fn stable_softplus_reference(
    runtime: &MlxRuntime,
    input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let zero_values = runtime.zeros(&input.shape(), input.dtype())?;
    let nonnegative_mask = runtime.greater_equal(input, &zero_values)?;
    let positive_part = runtime.where_select(&nonnegative_mask, input, &zero_values)?;
    let negative_input = runtime.negative(input)?;
    let negative_absolute_input =
        runtime.where_select(&nonnegative_mask, &negative_input, input)?;
    let exponentiated_decay = runtime.exp(&negative_absolute_input)?;
    let logarithmic_term = runtime.log1p(&exponentiated_decay)?;
    runtime.add(&positive_part, &logarithmic_term)
}
