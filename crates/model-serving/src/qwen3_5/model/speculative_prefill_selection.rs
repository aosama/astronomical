use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use super::Qwen3_5ExecutionError;

/// Selects sparse-prefill token positions without copying full scores to the CPU.
pub fn qwen3_5_select_speculative_prefill_token_positions_on_gpu(
    runtime: &MlxRuntime,
    importance_scores: &MlxArray,
    keep_percentage: u32,
    selection_chunk_token_count: usize,
    mandatory_trailing_token_count: usize,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let importance_score_shape = importance_scores.shape();
    if importance_score_shape.len() != 1 || importance_score_shape[0] <= 0 {
        return Err(invalid_selection(
            "importance scores must be a non-empty vector",
        ));
    }
    if importance_scores.dtype() != MlxDtype::Float32 {
        return Err(invalid_selection("importance scores must use float32"));
    }
    if !(1..=100).contains(&keep_percentage) {
        return Err(invalid_selection(
            "keep percentage must be between 1 and 100",
        ));
    }
    if selection_chunk_token_count == 0 || mandatory_trailing_token_count == 0 {
        return Err(invalid_selection(
            "selection and mandatory trailing token counts must be positive",
        ));
    }

    let importance_score_count = usize::try_from(importance_score_shape[0])
        .map_err(|_| invalid_selection("importance score count exceeds usize"))?;
    let selection_chunk_count = importance_score_count
        .checked_add(selection_chunk_token_count - 1)
        .ok_or_else(|| invalid_selection("selection chunk count overflowed"))?
        / selection_chunk_token_count;
    let retained_selection_chunk_count = usize::try_from(
        (u128::from(selection_chunk_count as u64) * u128::from(keep_percentage) + 99) / 100,
    )
    .map_err(|_| invalid_selection("retained selection chunk count overflowed"))?
    .max(1)
    .min(selection_chunk_count);
    let mandatory_trailing_start_position =
        importance_score_count.saturating_sub(mandatory_trailing_token_count);
    let first_mandatory_trailing_selection_chunk_index =
        mandatory_trailing_start_position / selection_chunk_token_count;
    let mandatory_trailing_selection_chunk_count = selection_chunk_count
        .checked_sub(first_mandatory_trailing_selection_chunk_index)
        .ok_or_else(|| invalid_selection("mandatory trailing chunk count underflowed"))?;
    let ranked_selection_chunk_end =
        selection_chunk_count - mandatory_trailing_selection_chunk_count;
    let ranked_selection_chunk_budget =
        retained_selection_chunk_count.saturating_sub(mandatory_trailing_selection_chunk_count);
    let selected_selection_chunk_count = ranked_selection_chunk_budget
        .checked_add(mandatory_trailing_selection_chunk_count)
        .ok_or_else(|| invalid_selection("selected selection chunk count overflowed"))?;

    let padded_importance_score_count = selection_chunk_count
        .checked_mul(selection_chunk_token_count)
        .ok_or_else(|| invalid_selection("padded importance score count overflowed"))?;
    let right_padding_score_count = padded_importance_score_count - importance_score_count;
    let padded_importance_scores = if right_padding_score_count == 0 {
        importance_scores.retain()?
    } else {
        let right_padding_scores = runtime.zeros(
            &[usize_to_i32(right_padding_score_count)?],
            MlxDtype::Float32,
        )?;
        runtime.concatenate_axis(&[importance_scores, &right_padding_scores], 0)?
    };
    let chunked_importance_scores = runtime.reshape(
        &padded_importance_scores,
        &[
            usize_to_i32(selection_chunk_count)?,
            usize_to_i32(selection_chunk_token_count)?,
        ],
    )?;
    let selection_chunk_score_sums = runtime.sum_axis(&chunked_importance_scores, 1, false)?;
    let selection_chunk_score_divisors = (0..selection_chunk_count)
        .map(|selection_chunk_index| {
            let selection_chunk_start = selection_chunk_index * selection_chunk_token_count;
            selection_chunk_token_count.min(importance_score_count - selection_chunk_start) as f32
        })
        .collect::<Vec<_>>();
    let selection_chunk_score_divisors = runtime.array_from_f32(
        &selection_chunk_score_divisors,
        &[usize_to_i32(selection_chunk_count)?],
    )?;
    let selection_chunk_scores =
        runtime.divide(&selection_chunk_score_sums, &selection_chunk_score_divisors)?;
    let ranked_selection_chunk_scores = runtime.slice(
        &selection_chunk_scores,
        &[0],
        &[usize_to_i32(ranked_selection_chunk_end)?],
        &[1],
    )?;

    let selected_ranked_selection_chunk_indices = if ranked_selection_chunk_budget == 0 {
        runtime.arange_i32(0, 0)?
    } else {
        let descending_selection_chunk_scores = runtime.negative(&ranked_selection_chunk_scores)?;
        let partitioned_selection_chunk_indices = runtime.argpartition_axis(
            &descending_selection_chunk_scores,
            usize_to_i32(ranked_selection_chunk_budget - 1)?,
            0,
        )?;
        let selected_unsorted_selection_chunk_indices = runtime.slice(
            &partitioned_selection_chunk_indices,
            &[0],
            &[usize_to_i32(ranked_selection_chunk_budget)?],
            &[1],
        )?;
        let selected_selection_chunk_sort_order =
            runtime.argsort_axis(&selected_unsorted_selection_chunk_indices, 0)?;
        let selected_sorted_selection_chunk_indices = runtime.take_axis(
            &selected_unsorted_selection_chunk_indices,
            &selected_selection_chunk_sort_order,
            0,
        )?;
        runtime.astype(&selected_sorted_selection_chunk_indices, MlxDtype::Int32)?
    };
    let mandatory_trailing_selection_chunk_indices = runtime.arange_i32(
        usize_to_i32(ranked_selection_chunk_end)?,
        usize_to_i32(selection_chunk_count)?,
    )?;
    let selected_selection_chunk_indices = runtime.concatenate_axis(
        &[
            &selected_ranked_selection_chunk_indices,
            &mandatory_trailing_selection_chunk_indices,
        ],
        0,
    )?;

    let selected_selection_chunk_indices =
        runtime.expand_dims(&selected_selection_chunk_indices, 1)?;
    let selection_chunk_token_count_scalar =
        runtime.array_from_i32(&[usize_to_i32(selection_chunk_token_count)?], &[])?;
    let selected_selection_chunk_starts = runtime.multiply(
        &selected_selection_chunk_indices,
        &selection_chunk_token_count_scalar,
    )?;
    let selection_chunk_token_offsets =
        runtime.arange_i32(0, usize_to_i32(selection_chunk_token_count)?)?;
    let selection_chunk_token_offsets = runtime.expand_dims(&selection_chunk_token_offsets, 0)?;
    let selected_token_positions = runtime.add(
        &selected_selection_chunk_starts,
        &selection_chunk_token_offsets,
    )?;
    let selected_token_positions = runtime.reshape(
        &selected_token_positions,
        &[usize_to_i32(
            selected_selection_chunk_count
                .checked_mul(selection_chunk_token_count)
                .ok_or_else(|| invalid_selection("selected token capacity overflowed"))?,
        )?],
    )?;
    let selected_token_count = selected_selection_chunk_count
        .checked_mul(selection_chunk_token_count)
        .and_then(|selected_token_capacity| {
            selected_token_capacity.checked_sub(right_padding_score_count)
        })
        .ok_or_else(|| invalid_selection("selected token count overflowed"))?;
    runtime
        .slice(
            &selected_token_positions,
            &[0],
            &[usize_to_i32(selected_token_count)?],
            &[1],
        )
        .map_err(Into::into)
}

fn usize_to_i32(amount: usize) -> Result<i32, Qwen3_5ExecutionError> {
    i32::try_from(amount).map_err(|_| invalid_selection("selection amount exceeds MLX range"))
}

fn invalid_selection(description: &'static str) -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput { description }
}
