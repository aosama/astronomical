//! The fp32 numerical oracle: the same projection through Accelerate's BLAS.
//!
//! Accelerate runs on the CPU in single precision with fp32 accumulation, so
//! its result is the yardstick both the Neural Engine (fp16) and MLX are
//! measured against.

use astronomical_neural_engine::program_shape::ProgramShape;

use thiserror::Error;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANSPOSE: i32 = 111;

unsafe extern "C" {
    // Accelerate's CBLAS single-precision matrix product (BLAS level 3).
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        row_count: i32,
        column_count: i32,
        inner_count: i32,
        alpha: f32,
        matrix_a: *const f32,
        leading_dimension_a: i32,
        matrix_b: *const f32,
        leading_dimension_b: i32,
        beta: f32,
        matrix_c: *mut f32,
        leading_dimension_c: i32,
    );
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReferenceProjectionError {
    #[error("expected {expected} weights, got {actual}")]
    WeightCountMismatch { expected: usize, actual: usize },
    #[error("expected {expected} activations, got {actual}")]
    ActivationCountMismatch { expected: usize, actual: usize },
}

/// `projected[o][t] = Σᵢ weights[o][i] · activations[i][t]`, all row-major.
pub fn reference_projection(
    shape: ProgramShape,
    weights: &[f32],
    activations: &[f32],
) -> Result<Vec<f32>, ReferenceProjectionError> {
    if weights.len() != shape.weight_element_count() {
        return Err(ReferenceProjectionError::WeightCountMismatch {
            expected: shape.weight_element_count(),
            actual: weights.len(),
        });
    }
    if activations.len() != shape.activation_element_count() {
        return Err(ReferenceProjectionError::ActivationCountMismatch {
            expected: shape.activation_element_count(),
            actual: activations.len(),
        });
    }
    let mut projected = vec![0.0_f32; shape.projected_element_count()];
    // ProgramShape guarantees every dimension fits in i32.
    let output_channels = shape.output_channels() as i32;
    let input_channels = shape.input_channels() as i32;
    let token_count = shape.token_count() as i32;
    // SAFETY: the three buffers are exactly M·K, K·N, and M·N elements long
    // (checked above), the leading dimensions equal the row-major row widths,
    // and Accelerate does not retain any pointer after returning.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANSPOSE,
            CBLAS_NO_TRANSPOSE,
            output_channels,
            token_count,
            input_channels,
            1.0,
            weights.as_ptr(),
            input_channels,
            activations.as_ptr(),
            token_count,
            0.0,
            projected.as_mut_ptr(),
            token_count,
        );
    }
    Ok(projected)
}

#[cfg(test)]
#[path = "accelerate_reference_tests.rs"]
mod tests;
