//! Thinking-allowance resolution against the caller's output budget.
//!
//! The thinking allowance is a ceiling on reasoning, not a floor: the model
//! may finish reasoning earlier through the natural transition. When the
//! caller's own output cap cannot hold the requested allowance plus the
//! transition reservation plus one visible answer token, the allowance
//! shrinks to fit instead of rejecting a satisfiable request (issue #652).
//! Pi compaction sends exactly that shape: an output cap of 16,000 with a
//! 16,384 high-effort allowance.

use super::thinking_budget::minimum_bounded_output_token_count;

/// Resolves the effective thinking allowance for one request.
///
/// Returns `None` when no positive allowance is active (thinking disabled or
/// a zero budget), the requested allowance when it already fits the output
/// budget, and the shrunken allowance otherwise. The caller logs the clamp
/// with the requested and effective values so the adjustment stays
/// attributable.
#[must_use]
pub fn resolve_effective_thinking_allowance(
    requested_thinking_budget: Option<u16>,
    enable_thinking: bool,
    max_output_tokens: u16,
    transition_token_count: usize,
) -> Option<u16> {
    let Some(thinking_budget) = requested_thinking_budget else {
        return None;
    };
    if !enable_thinking || thinking_budget == 0 {
        return None;
    }
    let fits_requested_allowance =
        minimum_bounded_output_token_count(thinking_budget, transition_token_count).is_some_and(
            |minimum_bounded_output_tokens| {
                usize::from(max_output_tokens) >= minimum_bounded_output_tokens
            },
        );
    if fits_requested_allowance {
        return Some(thinking_budget);
    }
    let allowance_room = usize::from(max_output_tokens).saturating_sub(transition_token_count);
    Some(u16::try_from(allowance_room.saturating_sub(1)).unwrap_or(u16::MAX))
}

/// The reservation the effective allowance demands from the output budget.
#[must_use]
pub fn effective_thinking_reservation_token_count(
    effective_thinking_allowance: Option<u16>,
    transition_token_count: usize,
) -> Option<usize> {
    let thinking_budget = effective_thinking_allowance?;
    minimum_bounded_output_token_count(thinking_budget, transition_token_count)
}
