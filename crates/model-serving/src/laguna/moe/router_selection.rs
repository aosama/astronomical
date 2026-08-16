//! CPU Laguna router: sigmoid top-K with selection-only correction bias.
//!
//! This is the authoritative score formula used by the GPU path. Direct-MLX
//! tests compare GPU selections against this owner for unique rankings.

use super::super::model::LagunaExecutionError;

/// Selected expert identities and the original sigmoid scores gathered for them.
#[derive(Clone, Debug, PartialEq)]
pub struct LagunaRouterSelection {
    expert_indices: Vec<u32>,
    original_scores: Vec<f32>,
    token_count: usize,
    experts_per_token: usize,
}

impl LagunaRouterSelection {
    /// Returns flattened selected expert ids, `token_count * experts_per_token` long.
    #[must_use]
    pub fn expert_indices(&self) -> &[u32] {
        &self.expert_indices
    }

    /// Returns original sigmoid scores aligned with [`Self::expert_indices`].
    #[must_use]
    pub fn original_scores(&self) -> &[f32] {
        &self.original_scores
    }

    /// Returns how many token rows were routed.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Returns the routed top-K width.
    #[must_use]
    pub const fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }
}

/// Selects top-K experts from row-major router logits `[token_count, expert_count]`.
///
/// Softcap, when positive, is applied to logits before sigmoid. Correction bias
/// is added only to the ranking scores. Gathered weights are the original
/// sigmoid values, optionally renormalized across the selected K.
pub fn select_laguna_router_experts(
    router_logits: &[f32],
    token_count: usize,
    expert_count: usize,
    experts_per_token: usize,
    correction_bias: Option<&[f32]>,
    router_logit_softcap: f64,
    normalizes_top_k_probabilities: bool,
) -> Result<LagunaRouterSelection, LagunaExecutionError> {
    validate_router_geometry(
        router_logits,
        token_count,
        expert_count,
        experts_per_token,
        correction_bias,
        router_logit_softcap,
    )?;
    let mut expert_indices = Vec::with_capacity(token_count * experts_per_token);
    let mut original_scores = Vec::with_capacity(token_count * experts_per_token);
    for token_index in 0..token_count {
        let logit_start = token_index * expert_count;
        let token_logits = &router_logits[logit_start..logit_start + expert_count];
        let mut ranked_experts = Vec::with_capacity(expert_count);
        for (expert_index, &logit) in token_logits.iter().enumerate() {
            let softcapped_logit = apply_router_logit_softcap(logit, router_logit_softcap);
            let original_score = sigmoid(softcapped_logit);
            let selection_score = original_score
                + correction_bias
                    .map(|bias_values| bias_values[expert_index])
                    .unwrap_or(0.0);
            ranked_experts.push((selection_score, expert_index as u32, original_score));
        }
        // Stable among ties: higher selection score first, then lower expert id.
        ranked_experts.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let mut selected_scores = Vec::with_capacity(experts_per_token);
        for &(_selection_score, expert_index, original_score) in
            ranked_experts.iter().take(experts_per_token)
        {
            expert_indices.push(expert_index);
            selected_scores.push(original_score);
        }
        if normalizes_top_k_probabilities {
            let selected_score_sum: f32 = selected_scores.iter().sum();
            if selected_score_sum > 0.0 {
                for selected_score in &mut selected_scores {
                    *selected_score /= selected_score_sum;
                }
            }
        }
        original_scores.extend(selected_scores);
    }
    Ok(LagunaRouterSelection {
        expert_indices,
        original_scores,
        token_count,
        experts_per_token,
    })
}

/// Applies `softcap * tanh(logit / softcap)` when the cap is positive.
#[must_use]
pub fn apply_router_logit_softcap(logit: f32, router_logit_softcap: f64) -> f32 {
    if router_logit_softcap > 0.0 {
        let softcap = router_logit_softcap as f32;
        (logit / softcap).tanh() * softcap
    } else {
        logit
    }
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

fn validate_router_geometry(
    router_logits: &[f32],
    token_count: usize,
    expert_count: usize,
    experts_per_token: usize,
    correction_bias: Option<&[f32]>,
    router_logit_softcap: f64,
) -> Result<(), LagunaExecutionError> {
    if token_count == 0
        || expert_count == 0
        || experts_per_token == 0
        || experts_per_token > expert_count
        || router_logits.len() != token_count.saturating_mul(expert_count)
        || router_logit_softcap.is_nan()
        || router_logit_softcap < 0.0
    {
        return Err(LagunaExecutionError::invalid_geometry(
            "Laguna router logits, top-K, and softcap must describe a valid selection",
        ));
    }
    if let Some(bias_values) = correction_bias
        && bias_values.len() != expert_count
    {
        return Err(LagunaExecutionError::invalid_geometry(
            "router correction bias must contain one value per expert",
        ));
    }
    Ok(())
}
