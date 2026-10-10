//! Qwen-owned hard reasoning-budget state.
//!
//! The controller advances only from tokens committed to the decoder. Keeping
//! forcing here prevents the public token stream from diverging from the model's
//! autoregressive history when a reasoning allowance is exhausted.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ThinkingBudgetPhase {
    VisibleAnswer,
    Thinking,
    ForcingTransition,
}

/// Request-local reasoning allowance and forced-transition cursor.
#[derive(Clone, Debug)]
pub struct Qwen3_5ThinkingBudgetState {
    thinking_budget: Option<u16>,
    thinking_token_count: u16,
    phase: ThinkingBudgetPhase,
    forced_transition_token_ids: Vec<u32>,
    natural_reasoning_end_token_ids: Vec<u32>,
    next_forced_transition_token_index: usize,
    forced_token_awaiting_commit: Option<u32>,
}

impl Qwen3_5ThinkingBudgetState {
    /// Creates a state whose model-owned transition must end at a recognized
    /// reasoning boundary whenever a positive hard allowance is active.
    pub fn new(
        starts_inside_thinking: bool,
        thinking_budget: Option<u16>,
        forced_transition_token_ids: Vec<u32>,
        natural_reasoning_end_token_ids: Vec<u32>,
    ) -> Result<Self, Qwen3_5ThinkingBudgetError> {
        let has_positive_thinking_budget = thinking_budget.is_some_and(|budget| budget > 0);
        let starts_inside_reasoning = starts_inside_thinking && !matches!(thinking_budget, Some(0));
        if starts_inside_thinking && has_positive_thinking_budget {
            let final_forced_token_id = forced_transition_token_ids
                .last()
                .ok_or(Qwen3_5ThinkingBudgetError::MissingForcedTransition)?;
            if !natural_reasoning_end_token_ids.contains(final_forced_token_id) {
                return Err(Qwen3_5ThinkingBudgetError::TransitionDoesNotEndReasoning);
            }
            if forced_transition_token_ids[..forced_transition_token_ids.len() - 1]
                .iter()
                .any(|token_id| natural_reasoning_end_token_ids.contains(token_id))
            {
                return Err(Qwen3_5ThinkingBudgetError::TransitionEndsReasoningEarly);
            }
        }
        let active_thinking_budget = if starts_inside_reasoning {
            thinking_budget
        } else {
            None
        };
        Ok(Self {
            thinking_budget: active_thinking_budget,
            thinking_token_count: 0,
            phase: if starts_inside_reasoning {
                ThinkingBudgetPhase::Thinking
            } else {
                ThinkingBudgetPhase::VisibleAnswer
            },
            forced_transition_token_ids,
            natural_reasoning_end_token_ids,
            next_forced_transition_token_index: 0,
            forced_token_awaiting_commit: None,
        })
    }

    /// Selects the next forced token before ordinary sampling. The caller must
    /// feed this exact token through the decoder before observing its commit.
    pub fn next_forced_transition_token_id(
        &mut self,
    ) -> Result<Option<u32>, Qwen3_5ThinkingBudgetError> {
        if self.phase != ThinkingBudgetPhase::ForcingTransition {
            return Ok(None);
        }
        if self.forced_token_awaiting_commit.is_some() {
            return Err(Qwen3_5ThinkingBudgetError::ForcedTokenNotCommitted);
        }
        let forced_token_id = self
            .forced_transition_token_ids
            .get(self.next_forced_transition_token_index)
            .copied()
            .ok_or(Qwen3_5ThinkingBudgetError::ForcedTransitionExhausted)?;
        self.forced_token_awaiting_commit = Some(forced_token_id);
        Ok(Some(forced_token_id))
    }

    /// Records the exact token committed to decoder history and returns whether
    /// its decoded text belongs to reasoning rather than the visible answer.
    pub fn observe_committed_token(
        &mut self,
        committed_token_id: u32,
    ) -> Result<bool, Qwen3_5ThinkingBudgetError> {
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
                        .ok_or(Qwen3_5ThinkingBudgetError::TokenCountOverflow)?;
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
                    .ok_or(Qwen3_5ThinkingBudgetError::ForcedTokenWasNotSelected)?;
                if committed_token_id != expected_forced_token_id {
                    return Err(Qwen3_5ThinkingBudgetError::ForcedTokenMismatch {
                        expected_token_id: expected_forced_token_id,
                        actual_token_id: committed_token_id,
                    });
                }
                self.next_forced_transition_token_index = self
                    .next_forced_transition_token_index
                    .checked_add(1)
                    .ok_or(Qwen3_5ThinkingBudgetError::TokenCountOverflow)?;
                let ends_reasoning = self.is_natural_reasoning_end(committed_token_id);
                if ends_reasoning {
                    self.phase = ThinkingBudgetPhase::VisibleAnswer;
                } else if self.next_forced_transition_token_index
                    >= self.forced_transition_token_ids.len()
                {
                    return Err(Qwen3_5ThinkingBudgetError::TransitionDoesNotEndReasoning);
                }
                Ok(!ends_reasoning)
            }
        }
    }

    #[must_use]
    pub const fn is_inside_thinking(&self) -> bool {
        !matches!(self.phase, ThinkingBudgetPhase::VisibleAnswer)
    }

    #[must_use]
    pub const fn is_forcing_transition(&self) -> bool {
        matches!(self.phase, ThinkingBudgetPhase::ForcingTransition)
    }

    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    #[must_use]
    pub const fn thinking_token_count(&self) -> u16 {
        self.thinking_token_count
    }

    fn is_natural_reasoning_end(&self, token_id: u32) -> bool {
        self.natural_reasoning_end_token_ids.contains(&token_id)
    }
}

/// A violated model-owned hard-budget state transition.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Qwen3_5ThinkingBudgetError {
    #[error("positive thinking budget requires a forced reasoning transition")]
    MissingForcedTransition,
    #[error("forced reasoning transition does not end at a recognized reasoning boundary")]
    TransitionDoesNotEndReasoning,
    #[error("forced reasoning transition contains a reasoning boundary before its final token")]
    TransitionEndsReasoningEarly,
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

/// Reserves the reasoning allowance, complete transition, and one visible token.
pub(in crate::qwen3_5) fn minimum_bounded_output_token_count(
    thinking_budget: u16,
    forced_transition_token_count: usize,
) -> Option<usize> {
    usize::from(thinking_budget)
        .checked_add(forced_transition_token_count)
        .and_then(|bounded_token_count| bounded_token_count.checked_add(1))
}
