#![allow(dead_code)]

use astronomical_mlx_c_rust::{MlxArray, MlxDtype};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxRuntimeError};

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
    actual_array: &astronomical_mlx_c_rust::MlxArray,
    expected_array: &astronomical_mlx_c_rust::MlxArray,
) {
    assert_eq!(actual_array.dtype(), MlxDtype::BFloat16);
    assert_eq!(expected_array.dtype(), MlxDtype::BFloat16);
    // BFloat16 widens exactly into float32 (top 16 bits, zero padding), so
    // comparing float32 bit patterns compares every bfloat16 bit. A tolerance
    // would accept an incorrect zero for values near the denormal range, where
    // a compiled-versus-reference divergence as large as the value itself
    // measures below any absolute threshold.
    let float32_actual_array = runtime
        .astype(actual_array, MlxDtype::Float32)
        .expect("the actual bfloat16 array should cast to float32");
    let float32_expected_array = runtime
        .astype(expected_array, MlxDtype::Float32)
        .expect("the expected bfloat16 array should cast to float32");
    let actual_bit_patterns: Vec<u32> = float32_actual_array
        .to_vec_f32()
        .expect("the actual array should evaluate")
        .iter()
        .map(|actual_float| actual_float.to_bits())
        .collect();
    let expected_bit_patterns: Vec<u32> = float32_expected_array
        .to_vec_f32()
        .expect("the expected array should evaluate")
        .iter()
        .map(|expected_float| expected_float.to_bits())
        .collect();
    assert_eq!(
        actual_bit_patterns.len(),
        expected_bit_patterns.len(),
        "the compared bfloat16 arrays must have the same element count"
    );
    for (element_index, (actual_bits, expected_bits)) in actual_bit_patterns
        .iter()
        .zip(expected_bit_patterns.iter())
        .enumerate()
    {
        assert_eq!(
            actual_bits, expected_bits,
            "bfloat16 element {element_index} must match the reference bit for bit"
        );
    }
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
