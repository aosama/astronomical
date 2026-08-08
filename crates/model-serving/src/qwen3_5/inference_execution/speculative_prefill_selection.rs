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
            Self::SelectionArithmeticOverflow => "speculative prefill selection arithmetic overflowed",
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
    let total_token_count = importance_scores.len();
    let mandatory_trailing_token_count = mandatory_trailing_token_count.min(total_token_count);
    let competing_token_count = total_token_count.saturating_sub(mandatory_trailing_token_count);
    let chunk_count = competing_token_count.div_ceil(selection_chunck_token_count);
    let mut chunk_averages = Vec::with_capacity(chunk_count);
    for chunk_index in 0..chunk_count {
        let chunk_start = chunk_index.saturating_mul(selection_chunck_token_count);
        let chunk_end = chunk_start
            .saturating_add(selection_chunck_token_count)
            .min(competing_token_count);
        if chunk_start >= chunk_end {
            break;
        }
        let mut chunk_sum = 0.0_f64;
        for score in &importance_scores[chunk_start..chunk_end] {
            chunk_sum += f64::from(*score);
        }
        let chunk_average = chunk_sum / (chunk_end.saturating_sub(chunk_start)) as f64;
        chunk_averages.push((chunk_average, chunk_index));
    }
    chunk_averages.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let keep_chunk_count = chunk_averages
        .len()
        .saturating_mul(usize::try_from(keep_percentage).unwrap_or(usize::MAX))
        .saturating_add(99)
        .saturating_div(100);
    let mut selected_positions = Vec::new();
    for (_, chunk_index) in chunk_averages.iter().take(keep_chunk_count) {
        let chunk_start = chunk_index.saturating_mul(selection_chunck_token_count);
        let chunk_end = chunk_start
            .saturating_add(selection_chunck_token_count)
            .min(competing_token_count);
        for position in chunk_start..chunk_end {
            if let Ok(position) = usize::try_from(position) {
                selected_positions.push(position);
            } else {
                return Err(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow);
            }
        }
    }
    for offset in 0..mandatory_trailing_token_count {
        let position = competing_token_count.saturating_add(offset);
        if let Ok(position) = usize::try_from(position) {
            selected_positions.push(position);
        } else {
            return Err(Qwen3_5SpeculativePrefillSelectionError::SelectionArithmeticOverflow);
        }
    }
    selected_positions.sort_unstable();
    selected_positions.dedup();
    Ok(selected_positions)
}

/// Merges speculative-prefill selected positions with image-pad positions.
///
/// Image-pad positions are inserted at their original indices before the
/// selected positions are adjusted to account for the visual embedding tokens.
pub fn qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
    selected_token_positions: Vec<usize>,
    image_pad_positions: &[usize],
    visual_embedding_token_count: usize,
) -> Vec<usize> {
    if image_pad_positions.is_empty() || visual_embedding_token_count == 0 {
        return selected_token_positions;
    }
    let mut merged_positions = Vec::with_capacity(
        selected_token_positions
            .len()
            .saturating_add(image_pad_positions.len()),
    );
    for selected_position in selected_token_positions {
        let mut adjusted_position = selected_position;
        for image_pad_position in image_pad_positions {
            if *image_pad_position <= selected_position {
                adjusted_position = adjusted_position.saturating_add(visual_embedding_token_count);
            } else {
                break;
            }
        }
        merged_positions.push(adjusted_position);
    }
    merged_positions.sort_unstable();
    merged_positions.dedup();
    merged_positions
}

/// Filters selected token positions to those within a half-open range.
pub fn qwen3_5_selected_speculative_prefill_positions_for_range(
    selected_token_positions: &[usize],
    range_start: usize,
    range_end: usize,
) -> Vec<usize> {
    selected_token_positions
        .iter()
        .copied()
        .filter(|position| *position >= range_start && *position < range_end)
        .collect()
}

/// Computes the draft scoring token range and selectable importance score count.
///
/// Returns `None` when the scoring plan is invalid (empty range or zero scores).
pub fn qwen3_5_speculative_prefill_scoring_plan(
    scoring_start_position_tokens: usize,
    final_prompt_index: usize,
    prompt_token_count: usize,
) -> Option<(std::ops::Range<usize>, usize)> {
    if scoring_start_position_tokens > prompt_token_count {
        return None;
    }
    let draft_scoring_end_position_tokens = if final_prompt_index >= prompt_token_count {
        prompt_token_count
    } else {
        final_prompt_index.saturating_add(1)
    };
    if draft_scoring_end_position_tokens <= scoring_start_position_tokens {
        return None;
    }
    let selectable_importance_score_count =
        draft_scoring_end_position_tokens.saturating_sub(scoring_start_position_tokens);
    if selectable_importance_score_count == 0 {
        return None;
    }
    Some((
        scoring_start_position_tokens..draft_scoring_end_position_tokens,
        selectable_importance_score_count,
    ))
}

/// Computes the selectable importance score range from draft scoring outcomes.
///
/// Returns `None` when the range is invalid or exceeds the available scores.
pub fn qwen3_5_speculative_prefill_selectable_importance_score_range(
    draft_importance_score_start_position_tokens: usize,
    scored_draft_prompt_token_count: usize,
    scoring_start_position_tokens: usize,
    selectable_importance_score_count: usize,
) -> Option<std::ops::Range<usize>> {
    let draft_scoring_end_position_tokens =
        draft_importance_score_start_position_tokens.saturating_add(scored_draft_prompt_token_count);
    if scoring_start_position_tokens < draft_importance_score_start_position_tokens {
        return None;
    }
    if scoring_start_position_tokens >= draft_scoring_end_position_tokens {
        return None;
    }
    let selectable_range_start =
        scoring_start_position_tokens.saturating_sub(draft_importance_score_start_position_tokens);
    let selectable_range_end = selectable_range_start.saturating_add(selectable_importance_score_count);
    if selectable_range_end > scored_draft_prompt_token_count {
        return None;
    }
    Some(selectable_range_start..selectable_range_end)
}
