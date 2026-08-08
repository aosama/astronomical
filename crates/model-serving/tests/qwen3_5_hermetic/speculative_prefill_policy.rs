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

/// Prompt-processing mode selected for one attempted Qwen3.5 chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5SpeculativePrefillChunckMode {
    /// Execute the ordinary target-only path for a request without an optional session.
    OrdinaryTarget,
    /// Execute only the target model before terminal additional-state capture.
    TargetOnlyPrefix,
    /// Retain target hidden rows and initialize private additional history once.
    TerminalAdditionalHistoryCapture,
}

/// Selects whether one prompt chunk may initialize private additional history.
#[must_use]
pub const fn qwen3_5_speculative_prefill_chunck_mode(
    has_optional_prediction_session: bool,
    prefill_end: usize,
    final_prompt_index: usize,
) -> Qwen3_5SpeculativePrefillChunckMode {
    if !has_optional_prediction_session {
        return Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget;
    }
    if prefill_end == final_prompt_index {
        Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
    } else {
        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix
    }
}

/// Selects sorted target prompt positions from per-token draft importance scores.
///
/// Scores are ranked by contiguous chunk average. The trailing window is removed
/// from ranked competition first and then always included in the result.
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
    let mandatory_trailing_chunk_count = if mandatory_trailing_token_count == 0 {
        0
    } else {
        mandatory_trailing_token_count
            .checked_add(selection_chunck_token_count - 1)
            .ok_or(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow)?
            / selection_chunck_token_count
    }
    .min(chunk_count);
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
        .map(|(chunk_index, _)| chunk_index)
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
    draft_selected_prompt_positions: Vec<usize>,
    image_pad_positions: &[usize],
    visual_embedding_token_count: usize,
) -> Vec<usize> {
    if image_pad_positions.is_empty() || visual_embedding_token_count == 0 {
        return draft_selected_prompt_positions;
    }

    let mut merged_positions = draft_selected_prompt_positions;
    merged_positions.reserve(image_pad_positions.len());
    merged_positions.extend_from_slice(image_pad_positions);
    merged_positions.sort_unstable();
    merged_positions.dedup();
    merged_positions
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
