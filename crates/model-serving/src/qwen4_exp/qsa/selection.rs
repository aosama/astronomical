//! Sparse-attention key selection for `qwen4_exp`.
//!
//! The full-attention layers of this family score keys with a low-rank
//! indexer projection and attend to at most the configured budget of them
//! per query. This owner is the selection rule: indexer scores in, ordered
//! key indices out, with causal masking applied before selection so a query
//! never selects a key it must not see.
//!
//! The selection is proven against the direct-MLX oracle's explicit full
//! scoring and the host-side top-k with stable tie handling; production
//! never certifies itself. Selection runs on the GPU through the same
//! partition-and-slice idiom the mixture-of-experts router uses, which the
//! routing contracts already pin.
//!
//! Masking uses a large finite negative sentinel rather than negative
//! infinity: Metal comparison behavior with infinite operands is not a
//! contract this owner may rely on, and a finite sentinel keeps every later
//! comparison exact.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

/// Selection geometry resolved from validated configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpSelectionPlan {
    /// Maximum keys any query may attend to.
    pub budget: u32,
}

impl Qwen4ExpSelectionPlan {
    /// Builds a plan from the validated sparse-attention configuration.
    #[must_use]
    pub const fn from_configuration(budget: u32) -> Self {
        Self { budget }
    }
}

/// Applies the causal mask to indexer scores on the GPU and selects the
/// top-`budget` keys per query.
///
/// Scores arrive shaped `[token_count, token_count]`: one row per query, one
/// column per key, in sequence order. Positions after the query are masked
/// before selection, so the causal contract is enforced inside this owner
/// and no caller can forget it. The returned indices are ordered by
/// descending score, with ties resolved toward the lower index exactly as
/// the host reference does.
///
/// When a query can see fewer keys than the budget, the remaining slots are
/// substituted with index zero, which every query can always see, so
/// attention stays well-defined; the parity contract pins this padding.
///
/// # Errors
/// When any MLX operation fails or the score shape is not square.
pub fn select_keys(
    runtime: &MlxRuntime,
    indexer_scores: &MlxArray,
    plan: &Qwen4ExpSelectionPlan,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<MlxArray, MlxRuntimeError> {
    performance_attribution
        .measure_operation(PerformanceOperation::Qwen4ExpIndexerSelection, |_| {
            select_keys_inner(runtime, indexer_scores, plan)
        })
}

fn select_keys_inner(
    runtime: &MlxRuntime,
    indexer_scores: &MlxArray,
    plan: &Qwen4ExpSelectionPlan,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = indexer_scores.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "qwen4_exp indexer selection",
            description: format!(
                "indexer scores must be [token_count, token_count], got {shape:?}"
            ),
        });
    }
    let token_count = shape[0];
    let budget = (plan.budget as i32).min(token_count);
    if budget <= 0 {
        // A zero budget selects nothing: the documented degenerate outcome,
        // returned as unsigned indices so downstream shapes stay uniform.
        return runtime.zeros(&[token_count, 0], MlxDtype::UInt32);
    }
    // Causal mask: query i may see keys j <= i. The comparison builds the
    // mask from positions, so the selection cannot see the future even if
    // the indexer scores a future key highly.
    let positions = runtime.arange_i32(0, token_count)?;
    let query_axis = runtime.expand_dims(&positions, 1)?;
    let key_axis = runtime.expand_dims(&positions, 0)?;
    // key <= query means visible: the future is key > query, negated into a
    // zero-or-one visibility mask with the comparison the runtime provides.
    let future = runtime.greater(&key_axis, &query_axis)?;
    let visible = runtime.subtract(&runtime.array_from_f32(&vec![1.0_f32], &[1])?, &future)?;
    let masked_sentinel = runtime.array_from_f32(&vec![-1.0e30_f32], &[1])?;
    let masked_sentinel = runtime.broadcast_to(&masked_sentinel, &[token_count, token_count])?;
    let masked = runtime.where_select(&visible, indexer_scores, &masked_sentinel)?;
    // Partition so the top budget entries land in the last positions, then
    // slice them: the same idiom the mixture-of-experts router pins. MLX's
    // partition places the largest values last, so the slice holds exactly
    // the top entries, ascending within the slice.
    let first_selected = token_count - budget;
    let partitioned_indices = runtime.argpartition_axis(&masked, first_selected, -1)?;
    let selected = runtime.slice(
        &partitioned_indices,
        &[0, first_selected],
        &[token_count, token_count],
        &[1, 1],
    )?;
    // When a query can see fewer keys than the budget, the partition fills
    // the remaining slots with masked positions whose order is
    // implementation-defined. Substitute those with index zero, which every
    // query can always see, so attention stays well-defined and the parity
    // contract against the host reference stays exact on the visible prefix.
    let gathered_scores = runtime.take_along_axis(&masked, &selected, -1)?;
    let gathered_shape = gathered_scores.shape();
    let substitution_floor = runtime.array_from_f32(&vec![-1.0e15_f32], &[1])?;
    let substitution_floor = runtime.broadcast_to(&substitution_floor, &gathered_shape)?;
    let above_floor = runtime.greater(&gathered_scores, &substitution_floor)?;
    let zeros = runtime.zeros(&selected.shape(), selected.dtype())?;
    let substituted = runtime.where_select(&above_floor, &selected, &zeros)?;
    // The partition slice orders its entries ascending by score, while the
    // contract promises descending order with ties toward the lower index.
    // Sort by the negated gathered scores and reorder the indices to match;
    // the padded zeros all share one score, so their relative order is
    // stable and irrelevant.
    let negated = runtime.multiply(
        &gathered_scores,
        &runtime.array_from_f32(&vec![-1.0_f32], &[1])?,
    )?;
    let descending_order = runtime.argsort_axis(&negated, -1)?;
    let ordered = runtime.take_along_axis(&substituted, &descending_order, -1)?;
    // The selection contract returns unsigned indices regardless of the
    // index dtype the partition produced.
    runtime.astype(&ordered, MlxDtype::UInt32)
}
