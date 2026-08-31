//! MLX-backed conversion from draft importance scores to absolute prompt positions.
//!
//! The pure selection formulas remain in `speculative_prefill_selection` so
//! hermetic tests can include them without constructing the complete Qwen runtime.
//! This adapter keeps slicing, ranking, and absolute-position offsetting on the
//! GPU until one final compact UInt32 vector is copied to the host.

/// Selects absolute prompt positions from the selectable slice of drafter scores.
///
/// The returned scoring outcome is passed through unchanged because it also owns
/// the reusable drafter prefix checkpoint needed by persistence. Consuming and
/// returning it makes that ownership transfer explicit without cloning MLX arrays.
pub(crate) fn select_absolute_speculative_prefill_positions_from_draft_scores(
    draft_model: &crate::qwen3_5::model::Qwen3_5Model,
    draft_scoring_outcome: crate::qwen3_5::model::Qwen3_5SpeculativePrefillDraftScoringOutcome,
    selectable_importance_score_range: std::ops::Range<usize>,
    scoring_start_position_tokens: usize,
    keep_percentage: u32,
    selection_chunk_token_count: u32,
    mandatory_trailing_token_count: u32,
) -> Result<
    (
        Vec<usize>,
        crate::qwen3_5::model::Qwen3_5SpeculativePrefillDraftScoringOutcome,
    ),
    crate::Qwen3_5ExecutionError,
> {
    // Scores must be one-dimensional and cover the requested half-open slice.
    // Validate shape before asking MLX to slice so malformed model output receives
    // a domain error rather than a lower-level indexing failure.
    let importance_score_shape = draft_scoring_outcome.importance_scores.shape();
    let selectable_importance_score_range_start_i32 = i32::try_from(
        selectable_importance_score_range.start,
    )
    .map_err(|_| crate::Qwen3_5ExecutionError::InvalidInput {
        description: "speculative-prefill importance score range start exceeds the MLX range",
    })?;
    let selectable_importance_score_range_end_i32 =
        i32::try_from(selectable_importance_score_range.end).map_err(|_| {
            crate::Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill importance score range end exceeds the MLX range",
            }
        })?;
    if importance_score_shape.len() != 1
        || importance_score_shape[0] < selectable_importance_score_range_end_i32
    {
        return Err(crate::Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill draft produced fewer importance scores than expected",
        });
    }
    let selectable_importance_scores = draft_model.runtime().slice(
        &draft_scoring_outcome.importance_scores,
        &[selectable_importance_score_range_start_i32],
        &[selectable_importance_score_range_end_i32],
        &[1],
    )?;
    // Ranking remains on GPU. The result is relative to the sliced selectable
    // score vector and is therefore not yet a prompt position.
    let selected_token_positions = crate::qwen3_5_select_speculative_prefill_token_positions_on_gpu(
        draft_model.runtime(),
        &selectable_importance_scores,
        keep_percentage,
        usize::try_from(selection_chunk_token_count).map_err(|_| {
            crate::Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill selection chunk count exceeds the usize range",
            }
        })?,
        usize::try_from(mandatory_trailing_token_count).map_err(|_| {
            crate::Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill trailing token count exceeds the usize range",
            }
        })?,
    )?;
    // GPU selection positions are relative to the sliced score vector. Offset
    // them on GPU before the one required host copy so callers receive absolute
    // prompt positions without an additional per-token host transformation.
    let scoring_start_position_scalar = draft_model.runtime().array_from_i32(
        &[i32::try_from(scoring_start_position_tokens).map_err(|_| {
            crate::Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill scoring start exceeds the MLX range",
            }
        })?],
        &[],
    )?;
    let absolute_selected_token_positions = draft_model
        .runtime()
        .add(&selected_token_positions, &scoring_start_position_scalar)?;
    // Persisted and host-side selection contracts use UInt32 absolute positions.
    // Cast before the sole host copy to avoid per-element GPU round trips.
    let absolute_selected_token_positions = draft_model.runtime().astype(
        &absolute_selected_token_positions,
        astronomical_runtime_integration::MlxDtype::UInt32,
    )?;
    let absolute_selected_token_positions = draft_model
        .runtime()
        .copy_u32_values(&absolute_selected_token_positions)?
        .into_iter()
        .map(|selected_token_position| {
            usize::try_from(selected_token_position).map_err(|_| {
                crate::Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill selected position exceeds usize",
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Preserve the scoring outcome for draft-prefix checkpoint persistence.
    Ok((absolute_selected_token_positions, draft_scoring_outcome))
}
