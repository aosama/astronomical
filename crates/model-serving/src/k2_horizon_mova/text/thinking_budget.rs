//! Hard IFM think-channel budget owned by K2 Horizon MoVA.
//!
//! The prompt always opens `<ifm|think>`. If the member never emits
//! `</ifm|think>`, the entire output budget stays in reasoning. This
//! controller injects the close token into decoder history so the
//! visible answer can start.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ThinkingBudgetPhase {
    VisibleAnswer,
    Thinking,
    ForcingTransition,
}

/// Request-local reasoning allowance and forced `</ifm|think>` cursor.
#[derive(Clone, Debug)]
pub struct K2HorizonMoVAThinkingBudgetState {
    thinking_budget: Option<u16>,
    thinking_token_count: u16,
    phase: ThinkingBudgetPhase,
    forced_transition_token_ids: Vec<u32>,
    natural_reasoning_end_token_ids: Vec<u32>,
    next_forced_transition_token_index: usize,
    forced_token_awaiting_commit: Option<u32>,
}

impl K2HorizonMoVAThinkingBudgetState {
    /// Builds budget state. A missing close-token sequence disables forcing.
    pub fn new(
        thinking_budget: Option<u16>,
        forced_transition_token_ids: Vec<u32>,
        natural_reasoning_end_token_ids: Vec<u32>,
    ) -> Result<Self, K2HorizonMoVAThinkingBudgetError> {
        let can_force = thinking_budget.is_some() && !forced_transition_token_ids.is_empty();
        if can_force {
            let final_forced_token_id = forced_transition_token_ids
                .last()
                .copied()
                .ok_or(K2HorizonMoVAThinkingBudgetError::MissingForcedTransition)?;
            if !natural_reasoning_end_token_ids.contains(&final_forced_token_id) {
                return Err(K2HorizonMoVAThinkingBudgetError::TransitionDoesNotEndReasoning);
            }
        }
        let starts_forcing = matches!(thinking_budget, Some(0)) && can_force;
        Ok(Self {
            thinking_budget: thinking_budget.filter(|_| can_force),
            thinking_token_count: 0,
            phase: if starts_forcing {
                ThinkingBudgetPhase::ForcingTransition
            } else if can_force {
                ThinkingBudgetPhase::Thinking
            } else {
                ThinkingBudgetPhase::Thinking
            },
            forced_transition_token_ids,
            natural_reasoning_end_token_ids,
            next_forced_transition_token_index: 0,
            forced_token_awaiting_commit: None,
        })
    }

    /// Selects the next forced close token before ordinary sampling.
    pub fn next_forced_transition_token_id(
        &mut self,
    ) -> Result<Option<u32>, K2HorizonMoVAThinkingBudgetError> {
        if self.phase != ThinkingBudgetPhase::ForcingTransition {
            return Ok(None);
        }
        if self.forced_token_awaiting_commit.is_some() {
            return Err(K2HorizonMoVAThinkingBudgetError::ForcedTokenNotCommitted);
        }
        let forced_token_id = self
            .forced_transition_token_ids
            .get(self.next_forced_transition_token_index)
            .copied()
            .ok_or(K2HorizonMoVAThinkingBudgetError::ForcedTransitionExhausted)?;
        self.forced_token_awaiting_commit = Some(forced_token_id);
        Ok(Some(forced_token_id))
    }

    /// Records the token committed to decoder history.
    ///
    /// Returns whether that token still belongs to the think channel.
    pub fn observe_committed_token(
        &mut self,
        committed_token_id: u32,
    ) -> Result<bool, K2HorizonMoVAThinkingBudgetError> {
        match self.phase {
            ThinkingBudgetPhase::VisibleAnswer => Ok(false),
            ThinkingBudgetPhase::Thinking => {
                if self.is_natural_reasoning_end(committed_token_id) {
                    self.phase = ThinkingBudgetPhase::VisibleAnswer;
                    return Ok(false);
                }
                if let Some(thinking_budget) = self.thinking_budget {
                    self.thinking_token_count = self
                        .thinking_token_count
                        .checked_add(1)
                        .ok_or(K2HorizonMoVAThinkingBudgetError::TokenCountOverflow)?;
                    if self.thinking_token_count >= thinking_budget {
                        self.phase = ThinkingBudgetPhase::ForcingTransition;
                    }
                }
                Ok(true)
            }
            ThinkingBudgetPhase::ForcingTransition => {
                let expected_forced_token_id = self
                    .forced_token_awaiting_commit
                    .take()
                    .ok_or(K2HorizonMoVAThinkingBudgetError::ForcedTokenWasNotSelected)?;
                if committed_token_id != expected_forced_token_id {
                    return Err(K2HorizonMoVAThinkingBudgetError::ForcedTokenMismatch {
                        expected_token_id: expected_forced_token_id,
                        actual_token_id: committed_token_id,
                    });
                }
                self.next_forced_transition_token_index = self
                    .next_forced_transition_token_index
                    .checked_add(1)
                    .ok_or(K2HorizonMoVAThinkingBudgetError::TokenCountOverflow)?;
                let ends_reasoning = self.is_natural_reasoning_end(committed_token_id);
                if ends_reasoning {
                    self.phase = ThinkingBudgetPhase::VisibleAnswer;
                } else if self.next_forced_transition_token_index
                    >= self.forced_transition_token_ids.len()
                {
                    return Err(K2HorizonMoVAThinkingBudgetError::TransitionDoesNotEndReasoning);
                }
                Ok(!ends_reasoning)
            }
        }
    }

    #[must_use]
    pub const fn is_inside_thinking(&self) -> bool {
        !matches!(self.phase, ThinkingBudgetPhase::VisibleAnswer)
    }
}

/// Resolves the family think-channel budget for one request.
///
/// `None` from REST still gets a K2 default so short Chat Completions
/// cannot spend every output token inside an unclosed think channel.
#[must_use]
pub fn resolve_k2_horizon_mova_thinking_budget(
    requested_thinking_budget: Option<u16>,
    maximum_output_tokens: u32,
    forced_transition_token_count: usize,
) -> Option<u16> {
    let reserved_after_think = u32::try_from(forced_transition_token_count.saturating_add(1))
        .unwrap_or(u32::MAX)
        .max(1);
    let maximum_think_tokens = maximum_output_tokens.saturating_sub(reserved_after_think);
    if maximum_think_tokens == 0 {
        return Some(0);
    }
    let uncapped_think_tokens = match requested_thinking_budget {
        Some(0) => return Some(0),
        Some(requested_thinking_budget) => u32::from(requested_thinking_budget),
        None => (maximum_output_tokens / 2).clamp(8, 256),
    };
    Some(u16::try_from(uncapped_think_tokens.min(maximum_think_tokens)).unwrap_or(u16::MAX))
}

fn is_natural_end(natural_reasoning_end_token_ids: &[u32], token_id: u32) -> bool {
    natural_reasoning_end_token_ids.contains(&token_id)
}

impl K2HorizonMoVAThinkingBudgetState {
    fn is_natural_reasoning_end(&self, token_id: u32) -> bool {
        is_natural_end(&self.natural_reasoning_end_token_ids, token_id)
    }
}

/// A violated K2 think-channel budget transition.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum K2HorizonMoVAThinkingBudgetError {
    #[error("positive thinking budget requires a forced reasoning transition")]
    MissingForcedTransition,
    #[error("forced reasoning transition does not end at a recognized reasoning boundary")]
    TransitionDoesNotEndReasoning,
    #[error("the previous forced reasoning token was not committed")]
    ForcedTokenNotCommitted,
    #[error("forced reasoning transition ended before its reasoning boundary")]
    ForcedTransitionExhausted,
    #[error("a model-selected token was committed while a forced reasoning token was required")]
    ForcedTokenWasNotSelected,
    #[error(
        "forced reasoning token mismatch: expected {expected_token_id}, received {actual_token_id}"
    )]
    ForcedTokenMismatch {
        expected_token_id: u32,
        actual_token_id: u32,
    },
    #[error("thinking-budget token counter overflowed")]
    TokenCountOverflow,
}
