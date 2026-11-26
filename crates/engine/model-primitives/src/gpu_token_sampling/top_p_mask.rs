//! The top-p (nucleus) mask over pre-selected top-k candidates, built from MLX
//! graph operations so it stays on the GPU.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

use super::{TokenSamplingError, sampling_failure, sampling_runtime_error};

#[cfg(test)]
#[path = "top_p_mask_tests.rs"]
mod tests;

/// The `[1, rows, top_k]` grid of candidates a top-p mask works over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateGrid {
    row_count: i32,
    top_k: i32,
}

impl CandidateGrid {
    /// Reads the grid from the shape of the candidate probabilities. Any other
    /// layout is refused because the mask broadcasts its constants across
    /// exactly `[1, rows, top_k]`.
    fn from_shape(selected_shape: &[i32]) -> Result<Self, TokenSamplingError> {
        if selected_shape.len() != 3 || selected_shape[0] != 1 || selected_shape[1] == 0 {
            return Err(sampling_failure(
                "selected probabilities must have shape [1, rows, top_k] with a positive row count",
            ));
        }
        Ok(Self {
            row_count: selected_shape[1],
            top_k: selected_shape[2],
        })
    }

    fn shape(self) -> [i32; 3] {
        [1, self.row_count, self.top_k]
    }

    fn element_count(self) -> usize {
        self.row_count as usize * self.top_k as usize
    }
}

/// At a top-p of zero, or of 1.0 and above, the mask keeps every candidate.
fn top_p_filters_nothing(top_p_thousandths: u16) -> bool {
    top_p_thousandths == 0 || top_p_thousandths >= 1_000
}

/// The candidates ordered from most to least probable: the sort permutation
/// (as `int32`, which `put_along_axis` requires) and the probabilities in that
/// order.
struct ProbabilityOrder {
    sorted_indices: MlxArray,
    sorted_probabilities: MlxArray,
}

/// Applies the top-p (nucleus) mask entirely on the GPU using MLX graph operations.
///
/// This applies the standard top-p boundary while operating on pre-selected top-k
/// candidates. The algorithm:
///
/// 1. Sort probabilities in descending order.
/// 2. Compute the exclusive cumulative mass of more-probable candidates.
/// 3. Map the cumulative sums back to the original (unsorted) order.
/// 4. Keep a candidate only while the preceding mass is below top-p.
pub fn apply_top_p_mask(
    runtime: &MlxRuntime,
    selected_probabilities: &MlxArray,
    selected_logits: &MlxArray,
    top_p_thousandths: u16,
) -> Result<MlxArray, TokenSamplingError> {
    let candidate_grid = CandidateGrid::from_shape(&selected_probabilities.shape())?;
    if top_p_filters_nothing(top_p_thousandths) {
        return logits_with_no_candidate_masked(runtime, selected_logits, candidate_grid.top_k);
    }

    let top_p = f32::from(top_p_thousandths) / 1_000.0;
    let probability_order = sort_by_descending_probability(runtime, selected_probabilities)?;
    let keep_mask_sorted = nucleus_keep_mask(
        runtime,
        &probability_order.sorted_probabilities,
        top_p,
        candidate_grid,
    )?;
    let masked_sorted_logits = mask_sorted_logits_outside_nucleus(
        runtime,
        selected_logits,
        &probability_order.sorted_indices,
        &keep_mask_sorted,
        candidate_grid,
    )?;
    let inverse_indices =
        inverse_sort_permutation(runtime, &probability_order.sorted_indices, candidate_grid)?;
    runtime
        .take_along_axis(&masked_sorted_logits, &inverse_indices, -1)
        .map_err(sampling_runtime_error)
}

fn logits_with_no_candidate_masked(
    runtime: &MlxRuntime,
    selected_logits: &MlxArray,
    top_k: i32,
) -> Result<MlxArray, TokenSamplingError> {
    // No top-p filtering needed. Use where_select with an always-true
    // condition to return a new lazy array that preserves the logits.
    let true_condition = runtime
        .array_from_f32(&[1.0], &[])
        .map_err(sampling_runtime_error)?;
    let true_broadcast = runtime
        .broadcast_to(&true_condition, &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    let neg_inf = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(sampling_runtime_error)?;
    let neg_inf_broadcast = runtime
        .broadcast_to(&neg_inf, &[1, 1, top_k])
        .map_err(sampling_runtime_error)?;
    runtime
        .where_select(&true_broadcast, selected_logits, &neg_inf_broadcast)
        .map_err(sampling_runtime_error)
}

fn sort_by_descending_probability(
    runtime: &MlxRuntime,
    selected_probabilities: &MlxArray,
) -> Result<ProbabilityOrder, TokenSamplingError> {
    // Sort probabilities in descending order by negating and argsort(ascending).
    let negated_probabilities = runtime
        .negative(selected_probabilities)
        .map_err(sampling_runtime_error)?;
    let sorted_indices = runtime
        .argsort_axis(&negated_probabilities, -1)
        .map_err(sampling_runtime_error)?;
    // Cast indices to int32 to ensure compatibility with put_along_axis.
    let sorted_indices_i32 = runtime
        .astype(&sorted_indices, MlxDtype::Int32)
        .map_err(sampling_runtime_error)?;
    let sorted_probabilities = runtime
        .take_along_axis(selected_probabilities, &sorted_indices_i32, -1)
        .map_err(sampling_runtime_error)?;
    Ok(ProbabilityOrder {
        sorted_indices: sorted_indices_i32,
        sorted_probabilities,
    })
}

fn nucleus_keep_mask(
    runtime: &MlxRuntime,
    sorted_probabilities: &MlxArray,
    top_p: f32,
    candidate_grid: CandidateGrid,
) -> Result<MlxArray, TokenSamplingError> {
    // A candidate survives while the probability mass of all more-probable
    // candidates is still below top-p. The exclusive scan keeps the candidate
    // that crosses the threshold in ascending-tail order.
    let preceding_probability_mass = runtime
        .cumsum(sorted_probabilities, -1, false, false)
        .map_err(sampling_runtime_error)?;

    // Build the mask in sorted order: keep where preceding mass < top-p.
    let top_p_constant = runtime
        .array_from_f32(&[top_p], &[])
        .map_err(sampling_runtime_error)?;
    let top_p_broadcast = runtime
        .broadcast_to(&top_p_constant, &candidate_grid.shape())
        .map_err(sampling_runtime_error)?;
    runtime
        .greater(&top_p_broadcast, &preceding_probability_mass)
        .map_err(sampling_runtime_error)
}

fn mask_sorted_logits_outside_nucleus(
    runtime: &MlxRuntime,
    selected_logits: &MlxArray,
    sorted_indices: &MlxArray,
    keep_mask_sorted: &MlxArray,
    candidate_grid: CandidateGrid,
) -> Result<MlxArray, TokenSamplingError> {
    // Gather logits in sorted order and apply the mask.
    let sorted_logits = runtime
        .take_along_axis(selected_logits, sorted_indices, -1)
        .map_err(sampling_runtime_error)?;
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .map_err(sampling_runtime_error)?;
    let neg_inf_broadcast = runtime
        .broadcast_to(&negative_infinity, &candidate_grid.shape())
        .map_err(sampling_runtime_error)?;
    runtime
        .where_select(keep_mask_sorted, &sorted_logits, &neg_inf_broadcast)
        .map_err(sampling_runtime_error)
}

fn inverse_sort_permutation(
    runtime: &MlxRuntime,
    sorted_indices: &MlxArray,
    candidate_grid: CandidateGrid,
) -> Result<MlxArray, TokenSamplingError> {
    // Scatter back to original order using inverse permutation.
    // inverse_indices[sorted_indices[i]] = i
    // The range pattern is row-independent: build one row and broadcast it to
    // every row instead of reshaping a flat arange (which only works when the
    // caller supplies exactly one row).
    let range_indices = runtime
        .arange_i32(0, candidate_grid.top_k)
        .map_err(sampling_runtime_error)?;
    let range_row = runtime
        .reshape(&range_indices, &[1, 1, candidate_grid.top_k])
        .map_err(sampling_runtime_error)?;
    let range_indices_reshaped = runtime
        .broadcast_to(&range_row, &candidate_grid.shape())
        .map_err(sampling_runtime_error)?;
    let zeros_template = runtime
        .array_from_i32(
            &vec![0_i32; candidate_grid.element_count()],
            &candidate_grid.shape(),
        )
        .map_err(sampling_runtime_error)?;
    runtime
        .put_along_axis(&zeros_template, sorted_indices, &range_indices_reshaped, -1)
        .map_err(sampling_runtime_error)
}
