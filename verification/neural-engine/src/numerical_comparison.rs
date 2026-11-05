//! Error metrics between a reference result and a candidate engine's result.
//!
//! Issue #161 asks for these exact metrics so every rung reports the same
//! evidence: largest absolute and relative error, mean absolute error, cosine
//! similarity, and how many candidate values are not finite.
//!
//! Element-wise relative error proved misleading for projections: an output
//! that happens to sit just above the floor below turns an ordinary fp16
//! rounding step into a "77 percent" error (first Neural Engine run). The
//! largest absolute error divided by the reference's root-mean-square
//! magnitude is reported alongside it as the scale-aware worst case.

use thiserror::Error;

// Relative error is undefined near zero; below this magnitude the reference
// value is treated as zero and only absolute error is meaningful.
const RELATIVE_ERROR_FLOOR: f64 = 1.0e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericalComparison {
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
    /// `maximum_absolute_error / rms(reference)`.
    pub maximum_error_relative_to_reference_rms: f64,
    pub mean_absolute_error: f64,
    pub cosine_similarity: f64,
    pub non_finite_candidate_count: usize,
    pub compared_value_count: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NumericalComparisonError {
    #[error("reference has {reference_count} values but candidate has {candidate_count}")]
    LengthMismatch {
        reference_count: usize,
        candidate_count: usize,
    },
    #[error("there are no values to compare")]
    Empty,
}

#[derive(Default)]
struct RunningTotals {
    maximum_absolute_error: f64,
    maximum_relative_error: f64,
    absolute_error_sum: f64,
    dot_product: f64,
    reference_squared_sum: f64,
    candidate_squared_sum: f64,
    non_finite_candidate_count: usize,
}

impl RunningTotals {
    fn accumulate(&mut self, reference_value: f64, candidate_value: f64) {
        if !candidate_value.is_finite() {
            self.non_finite_candidate_count += 1;
            return;
        }
        let absolute_error = (reference_value - candidate_value).abs();
        self.maximum_absolute_error = self.maximum_absolute_error.max(absolute_error);
        if reference_value.abs() >= RELATIVE_ERROR_FLOOR {
            let relative_error = absolute_error / reference_value.abs();
            self.maximum_relative_error = self.maximum_relative_error.max(relative_error);
        }
        self.absolute_error_sum += absolute_error;
        self.dot_product += reference_value * candidate_value;
        self.reference_squared_sum += reference_value * reference_value;
        self.candidate_squared_sum += candidate_value * candidate_value;
    }
}

pub fn compare_values(
    reference_values: &[f32],
    candidate_values: &[f32],
) -> Result<NumericalComparison, NumericalComparisonError> {
    if reference_values.len() != candidate_values.len() {
        return Err(NumericalComparisonError::LengthMismatch {
            reference_count: reference_values.len(),
            candidate_count: candidate_values.len(),
        });
    }
    if reference_values.is_empty() {
        return Err(NumericalComparisonError::Empty);
    }
    let mut totals = RunningTotals::default();
    for (reference_value, candidate_value) in reference_values.iter().zip(candidate_values) {
        totals.accumulate(f64::from(*reference_value), f64::from(*candidate_value));
    }
    let norm_product = (totals.reference_squared_sum * totals.candidate_squared_sum).sqrt();
    let reference_rms = (totals.reference_squared_sum / reference_values.len() as f64).sqrt();
    Ok(NumericalComparison {
        maximum_absolute_error: totals.maximum_absolute_error,
        maximum_relative_error: totals.maximum_relative_error,
        maximum_error_relative_to_reference_rms: rms_normalized(
            totals.maximum_absolute_error,
            reference_rms,
        ),
        mean_absolute_error: totals.absolute_error_sum / reference_values.len() as f64,
        // Two all-zero vectors are identical, which cosine similarity
        // otherwise leaves as 0/0.
        cosine_similarity: if norm_product == 0.0 {
            1.0
        } else {
            totals.dot_product / norm_product
        },
        non_finite_candidate_count: totals.non_finite_candidate_count,
        compared_value_count: reference_values.len(),
    })
}

fn rms_normalized(maximum_absolute_error: f64, reference_rms: f64) -> f64 {
    if reference_rms > 0.0 {
        maximum_absolute_error / reference_rms
    } else if maximum_absolute_error == 0.0 {
        0.0
    } else {
        // Any error against an all-zero reference is unbounded in scale.
        f64::INFINITY
    }
}

#[cfg(test)]
#[path = "numerical_comparison_tests.rs"]
mod tests;
