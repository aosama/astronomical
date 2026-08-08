/// Failure while selecting target prompt positions from draft importance scores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5SpeculativePrefillSelectionError {
    EmptyImportanceScores,
    KeepPercentageOutOfRange,
    SelectionChunckTokenCountMustBePositive,
    ImportanceScoreNotFinite,
    SelectionArithmeticOverflow,
}

impl std::fmt::Display for Qwen3_5SpeculativePrefillSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self {
            Self::EmptyImportanceScores => "importance scores must not be empty",
            Self::KeepPercentageOutOfRange => "keep percentage must be between 1 and 100",
            Self::SelectionChunckTokenCountMustBePositive => {
                "selection chunk token count must be positive"
            }
            Self::ImportanceScoreNotFinite => "importance scores must be finite",
            Self::SelectionArithmeticOverflow => {
                "speculative prefill selection arithmetic overflowed"
            }
        };
        formatter.write_str(description)
    }
}

impl std::error::Error for Qwen3_5SpeculativePrefillSelectionError {}

/// Selects sorted target prompt positions from per-token draft importance scores.
///
/// Scores are ranked by contiguous chunk average. The trailing window is removed
/// from the ranked competition first and then always included in the result.
pub fn qwen3_5_select_speculative_prefill_token_positions(
    importance_scores: &[f32],
    keep_percentage: u32,
    selection_chunck_token_count: usize,
    mandatory_trailing_token_count: usize,
) -> Result<Vec<usize>, Qwen3_5SpeculativePrefillSelectionError> {
    if importance_scores.is_empty() {
        return Err(Qwen3_5SpeculativePrefillSelectionError::EmptyImportanceScores);
    }
    if !(1..=100).contains(&keep_percentage) {
        return Err(Qwen3_5SpeculativePrefillSelectionError::KeepPercentageOutOfRange);
    }
    if selection_chunck_token_count == 0 {
        return Err(
            Qwen3_5SpeculativePrefillSelectionError::SelectionChunckTokenCountMustBePositive,
        );
    }
    if importance_scores.iter().any(|score| !score.is_finite()) {
        return Err(Qwen3_5SpeculativePrefillSelectionError::ImportanceScoreNotFinite);
    }

    let chunk_count = importance_scores
        .len()
        .checked_add(selection_chunck_token_count - 1)
        .ok_or(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow)?
        / selection_chunck_token_count;
    let retained_chunk_count =
        usize::try_from((u128::from(chunk_count as u64) * u128::from(keep_percentage) + 99) / 100)
            .map_err(|_| Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow)?
            .max(1)
            .min(chunk_count);
    let mandatory_trailing_start_position = importance_scores
        .len()
        .saturating_sub(mandatory_trailing_token_count);
    let first_mandatory_trailing_chunk_index = if mandatory_trailing_token_count == 0 {
        chunk_count
    } else {
        mandatory_trailing_start_position / selection_chunck_token_count
    };
    let mandatory_trailing_chunk_count = chunk_count - first_mandatory_trailing_chunk_index;
    let ranked_chunk_end = chunk_count - mandatory_trailing_chunk_count;
    let ranked_chunk_budget = retained_chunk_count.saturating_sub(mandatory_trailing_chunk_count);

    let mut ranked_chunks = Vec::with_capacity(ranked_chunk_end);
    for chunk_index in 0..ranked_chunk_end {
        let chunk_start = chunk_index
            .checked_mul(selection_chunck_token_count)
            .ok_or(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow)?;
        let chunk_end = chunk_start
            .saturating_add(selection_chunck_token_count)
            .min(importance_scores.len());
        let chunk_score = importance_scores[chunk_start..chunk_end]
            .iter()
            .copied()
            .sum::<f32>()
            / (chunk_end - chunk_start) as f32;
        if !chunk_score.is_finite() {
            return Err(Qwen3_5SpeculativePrefillSelectionError::ImportanceScoreNotFinite);
        }
        ranked_chunks.push((chunk_index, chunk_score));
    }
    ranked_chunks.sort_by(|left_chunk, right_chunk| {
        right_chunk
            .1
            .total_cmp(&left_chunk.1)
            .then_with(|| left_chunk.0.cmp(&right_chunk.0))
    });

    let mut selected_chunk_indices = ranked_chunks
        .into_iter()
        .take(ranked_chunk_budget)
        .map(|(chunk_index, _chunk_score)| chunk_index)
        .collect::<Vec<_>>();
    selected_chunk_indices.extend(ranked_chunk_end..chunk_count);
    selected_chunk_indices.sort_unstable();

    let mut selected_token_positions = Vec::new();
    for chunk_index in selected_chunk_indices {
        let chunk_start = chunk_index
            .checked_mul(selection_chunck_token_count)
            .ok_or(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow)?;
        let chunk_end = chunk_start
            .saturating_add(selection_chunck_token_count)
            .min(importance_scores.len());
        selected_token_positions.extend(chunk_start..chunk_end);
    }
    Ok(selected_token_positions)
}

/// Retains every selectable image-pad location so sparse target prefill receives
/// the complete ordered visual representation for each image.
pub fn qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
    mut draft_selected_prompt_positions: Vec<usize>,
    prompt_token_ids: &[u32],
    selectable_prompt_start_position: usize,
    selectable_prompt_end_position: usize,
    image_pad_token_id: u32,
) -> Result<Vec<usize>, Qwen3_5SpeculativePrefillSelectionError> {
    if selectable_prompt_start_position > selectable_prompt_end_position
        || selectable_prompt_end_position > prompt_token_ids.len()
    {
        return Err(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow);
    }
    if draft_selected_prompt_positions
        .iter()
        .any(|draft_selected_prompt_position| {
            *draft_selected_prompt_position >= selectable_prompt_end_position
        })
    {
        return Err(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow);
    }
    for (prompt_token_position, prompt_token_id) in prompt_token_ids
        .iter()
        .enumerate()
        .take(selectable_prompt_end_position)
        .skip(selectable_prompt_start_position)
    {
        if *prompt_token_id == image_pad_token_id {
            draft_selected_prompt_positions.push(prompt_token_position);
        }
    }
    draft_selected_prompt_positions.sort_unstable();
    draft_selected_prompt_positions.dedup();
    Ok(draft_selected_prompt_positions)
}

/// Returns selected absolute prompt positions that belong to one prompt chunk.
#[must_use]
pub fn qwen3_5_selected_speculative_prefill_positions_for_range(
    selected_token_positions: &[usize],
    prefill_start: usize,
    prefill_end: usize,
) -> Vec<usize> {
    if prefill_start >= prefill_end {
        return Vec::new();
    }
    let first_selected_position = selected_token_positions
        .partition_point(|selected_token_position| *selected_token_position < prefill_start);
    let first_position_after_chunk = selected_token_positions
        .partition_point(|selected_token_position| *selected_token_position < prefill_end);
    selected_token_positions[first_selected_position..first_position_after_chunk].to_vec()
}

/// Plans complete-prompt draft scoring while reserving the final prompt token
/// for the target model's generation-kickoff forward pass.
#[must_use]
pub fn qwen3_5_speculative_prefill_scoring_plan(
    scoring_start_position_tokens: usize,
    final_prompt_index: usize,
    prompt_token_count: usize,
) -> Option<(std::ops::Range<usize>, usize)> {
    if final_prompt_index.checked_add(1)? != prompt_token_count
        || scoring_start_position_tokens >= final_prompt_index
    {
        return None;
    }

    Some((
        scoring_start_position_tokens..prompt_token_count,
        final_prompt_index - scoring_start_position_tokens,
    ))
}

/// Maps the target's uncached selectable prompt range into the draft score
/// vector, which can begin before the target's restored prefix.
#[must_use]
pub fn qwen3_5_speculative_prefill_selectable_importance_score_range(
    draft_scoring_start_position_tokens: usize,
    scored_draft_prompt_token_count: usize,
    target_selection_start_position_tokens: usize,
    selectable_importance_score_count: usize,
) -> Option<std::ops::Range<usize>> {
    let target_selection_end_position_tokens =
        target_selection_start_position_tokens.checked_add(selectable_importance_score_count)?;
    let draft_scoring_end_position_tokens =
        draft_scoring_start_position_tokens.checked_add(scored_draft_prompt_token_count)?;
    if target_selection_start_position_tokens < draft_scoring_start_position_tokens
        || target_selection_end_position_tokens > draft_scoring_end_position_tokens
    {
        return None;
    }

    Some(
        target_selection_start_position_tokens.checked_sub(draft_scoring_start_position_tokens)?
            ..target_selection_end_position_tokens
                .checked_sub(draft_scoring_start_position_tokens)?,
    )
}
