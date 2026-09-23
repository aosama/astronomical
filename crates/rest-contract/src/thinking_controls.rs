//! Unified resolution of the client spellings that control the thinking budget.
//!
//! Clients trained on different gateways express the same control in several
//! shapes: the OpenAI `reasoning_effort` level and numeric budget aliases, the
//! OpenRouter-style `reasoning` object, a flat `enable_thinking` flag, and the
//! vLLM-style `chat_template_kwargs` block. This module is the single place
//! that resolves every spelling into the worker's one hard thinking budget
//! plus the stream exclusion preference, and fails loudly on any contradiction
//! so a caller bug never silently changes how much a model thinks.

use serde::Deserialize;
use thiserror::Error;

/// OpenRouter-style `reasoning` object accepted on both generation endpoints.
///
/// Unknown fields are absorbed rather than rejected: this object is
/// provider-shaped and new spellings appear weekly (issue #777), so only the
/// contradictory *control values* below are caller errors.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ReasoningRequestObject {
    /// Effort level name; see `reasoning_effort_level` for the accepted set.
    #[serde(default)]
    pub effort: Option<String>,
    /// Direct thinking-token limit (Anthropic-style), equivalent to the
    /// top-level numeric budget spellings.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Explicit on/off switch for the thinking channel.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Keep the model thinking but withhold reasoning deltas from the stream.
    #[serde(default)]
    pub exclude: Option<bool>,
    /// OpenAI summary-inclusion preference (`auto`/`concise`/`detailed`).
    /// Absorbed without behavior change: Astronomical always streams
    /// reasoning deltas unless `exclude` withholds them.
    #[serde(default)]
    pub summary: Option<String>,
}

/// vLLM-style template-kwarg block; only the thinking toggle is meaningful here.
/// Unknown template variables are absorbed: gateways add template-specific
/// kwargs that are not thinking controls.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatTemplateKwargsRequestObject {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

/// The one resolved thinking control set consumed by translations and streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThinkingControls {
    /// Worker hard budget. `Some(0)` is an explicit disable that renders the
    /// thinking channel closed; `None` leaves the model's default in force.
    pub budget: Option<u32>,
    /// Whether reasoning deltas must be withheld from the response stream.
    pub reasoning_excluded: bool,
}

/// Every submitted thinking-control spelling, in endpoint-field order.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThinkingControlsInputs<'a> {
    pub thinking_budget: Option<u32>,
    pub thinking_token_budget: Option<u32>,
    pub thinking_budget_tokens: Option<u32>,
    pub reasoning: Option<&'a ReasoningRequestObject>,
    pub reasoning_effort: Option<&'a str>,
    pub enable_thinking: Option<bool>,
    pub chat_template_kwargs: Option<&'a ChatTemplateKwargsRequestObject>,
}

impl<'a> ThinkingControlsInputs<'a> {
    /// Resolves all spellings into one budget plus the exclusion preference.
    ///
    /// Precedence: explicit numbers beat level names, and an explicit disable
    /// (`off`/`none` level or any false flag) beats level names. Numbers
    /// disagreeing with each other, levels disagreeing with each other, flags
    /// disagreeing with each other, or a disable combined with a positive
    /// budget are caller bugs that fail loudly.
    pub fn resolve(self) -> Result<ThinkingControls, ThinkingControlsError> {
        let reasoning = self.reasoning.unwrap_or(&ReasoningRequestObject {
            effort: None,
            max_tokens: None,
            enabled: None,
            exclude: None,
            summary: None,
        });

        let resolved_numeric = resolve_numeric_group(
            [
                self.thinking_budget,
                self.thinking_token_budget,
                self.thinking_budget_tokens,
                reasoning.max_tokens,
            ],
            self,
        )?;
        let level_budget = resolve_level_group(self.reasoning_effort, reasoning.effort.as_deref())?;
        let thinking_enabled_by_flags = resolve_enable_flag_group(self, reasoning.enabled)?;

        let positive_numeric_budget = resolved_numeric.filter(|numeric_budget| *numeric_budget > 0);
        let positive_level_budget = match level_budget {
            Some(ResolvedEffort::Budget(budget)) => Some(budget),
            Some(ResolvedEffort::Disabled) | None => None,
        };
        // Explicit numbers beat level names.
        let positive_budget = positive_numeric_budget.or(positive_level_budget);
        // A zero budget, any false flag, or an off/none level disables thinking.
        let disabled = matches!(resolved_numeric, Some(0))
            || !thinking_enabled_by_flags
            || matches!(level_budget, Some(ResolvedEffort::Disabled));

        let budget = if disabled {
            if let Some(thinking_budget) = positive_budget {
                return Err(
                    ThinkingControlsError::ThinkingDisabledWhileBudgetRequested { thinking_budget },
                );
            }
            Some(0)
        } else {
            positive_budget
        };
        Ok(ThinkingControls {
            budget,
            reasoning_excluded: reasoning.exclude.unwrap_or(false),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResolvedEffort {
    Budget(u32),
    Disabled,
}

fn resolve_numeric_group(
    numeric_spellings: [Option<u32>; 4],
    inputs: ThinkingControlsInputs<'_>,
) -> Result<Option<u32>, ThinkingControlsError> {
    let mut resolved: Option<u32> = None;
    for numeric_spelling in numeric_spellings {
        let Some(numeric_spelling) = numeric_spelling else {
            continue;
        };
        match resolved {
            None => resolved = Some(numeric_spelling),
            Some(agreed_budget) if agreed_budget == numeric_spelling => {}
            Some(_) => {
                return Err(ThinkingControlsError::ConflictingNumericThinkingBudgets {
                    thinking_budget: inputs.thinking_budget,
                    thinking_token_budget: inputs.thinking_token_budget,
                    thinking_budget_tokens: inputs.thinking_budget_tokens,
                    reasoning_max_tokens: inputs
                        .reasoning
                        .and_then(|reasoning| reasoning.max_tokens),
                });
            }
        }
    }
    Ok(resolved)
}

fn resolve_level_group(
    reasoning_effort: Option<&str>,
    reasoning_object_effort: Option<&str>,
) -> Result<Option<ResolvedEffort>, ThinkingControlsError> {
    let top_level_effort = reasoning_effort.map(reasoning_effort_level).transpose()?;
    let nested_effort = reasoning_object_effort
        .map(reasoning_effort_level)
        .transpose()?;
    match (top_level_effort, nested_effort) {
        (None, None) => Ok(None),
        (Some(top_level_effort), Some(nested_effort)) => {
            if matches!(
                (top_level_effort, nested_effort),
                (ResolvedEffort::Budget(actual), ResolvedEffort::Budget(expected))
                    if actual != expected
            ) {
                return Err(ThinkingControlsError::ConflictingReasoningEfforts {
                    reasoning_effort: reasoning_effort.map(String::from).unwrap_or_default(),
                    reasoning_object_effort: reasoning_object_effort
                        .map(String::from)
                        .unwrap_or_default(),
                });
            }
            // A disable in either spelling outranks a level name in the other.
            let resolved_effort = if matches!(top_level_effort, ResolvedEffort::Disabled)
                || matches!(nested_effort, ResolvedEffort::Disabled)
            {
                ResolvedEffort::Disabled
            } else {
                top_level_effort
            };
            return Ok(Some(resolved_effort));
        }
        (Some(top_level_effort), None) => Ok(Some(top_level_effort)),
        (None, Some(nested_effort)) => Ok(Some(nested_effort)),
    }
}

fn resolve_enable_flag_group(
    inputs: ThinkingControlsInputs<'_>,
    reasoning_enabled: Option<bool>,
) -> Result<bool, ThinkingControlsError> {
    let flag_values = [
        reasoning_enabled,
        inputs.enable_thinking,
        inputs
            .chat_template_kwargs
            .and_then(|chat_template_kwargs| chat_template_kwargs.enable_thinking),
    ];
    let mut thinking_enabled = true;
    let mut flags_observed = 0usize;
    for flag_value in flag_values {
        let Some(flag_value) = flag_value else {
            continue;
        };
        flags_observed += 1;
        if flags_observed > 1 && flag_value != thinking_enabled {
            return Err(ThinkingControlsError::ConflictingThinkingEnableFlags {
                reasoning_enabled,
                enable_thinking: inputs.enable_thinking,
                chat_template_enable_thinking: inputs
                    .chat_template_kwargs
                    .and_then(|chat_template_kwargs| chat_template_kwargs.enable_thinking),
            });
        }
        thinking_enabled = flag_value;
    }
    Ok(thinking_enabled)
}

/// Maps an effort level name to the thinking-token budget coding agents
/// display for that level. The values mirror Pi's default thinking budgets so
/// the number an agent shows its user is the number this server enforces.
/// `xhigh` and `max` clamp to `high`, matching the agents' own clamping, and
/// `off`/`none` disable the thinking channel entirely.
fn reasoning_effort_level(reasoning_effort: &str) -> Result<ResolvedEffort, ThinkingControlsError> {
    match reasoning_effort {
        "minimal" => Ok(ResolvedEffort::Budget(1024)),
        "low" => Ok(ResolvedEffort::Budget(2048)),
        "medium" => Ok(ResolvedEffort::Budget(8192)),
        "high" | "xhigh" | "max" => Ok(ResolvedEffort::Budget(16384)),
        "off" | "none" => Ok(ResolvedEffort::Disabled),
        other => Err(ThinkingControlsError::UnknownReasoningEffort {
            reasoning_effort: other.to_owned(),
        }),
    }
}

/// A contradictory or unrecognized thinking-control spelling, rejected before
/// worker admission.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ThinkingControlsError {
    /// Two or more numeric budget spellings were supplied and disagreed.
    #[error(
        "thinking_budget ({thinking_budget:?}) conflicts with thinking_token_budget ({thinking_token_budget:?}), thinking_budget_tokens ({thinking_budget_tokens:?}), and reasoning.max_tokens ({reasoning_max_tokens:?})"
    )]
    ConflictingNumericThinkingBudgets {
        thinking_budget: Option<u32>,
        thinking_token_budget: Option<u32>,
        thinking_budget_tokens: Option<u32>,
        reasoning_max_tokens: Option<u32>,
    },
    /// Two level names mapped to different thinking budgets.
    #[error(
        "reasoning_effort '{reasoning_effort}' conflicts with reasoning.effort '{reasoning_object_effort}'"
    )]
    ConflictingReasoningEfforts {
        reasoning_effort: String,
        reasoning_object_effort: String,
    },
    /// An effort level name this server cannot enforce.
    #[error(
        "reasoning_effort '{reasoning_effort}' is not a recognized thinking level; expected minimal, low, medium, high, xhigh, max, off, or none"
    )]
    UnknownReasoningEffort { reasoning_effort: String },
    /// Thinking enable flags were supplied and disagreed.
    #[error(
        "thinking enable flags disagree: reasoning.enabled ({reasoning_enabled:?}), enable_thinking ({enable_thinking:?}), chat_template_kwargs.enable_thinking ({chat_template_enable_thinking:?})"
    )]
    ConflictingThinkingEnableFlags {
        reasoning_enabled: Option<bool>,
        enable_thinking: Option<bool>,
        chat_template_enable_thinking: Option<bool>,
    },
    /// The client disabled thinking while also requesting a positive budget.
    #[error(
        "thinking is disabled while a positive thinking budget of {thinking_budget} tokens is requested"
    )]
    ThinkingDisabledWhileBudgetRequested { thinking_budget: u32 },
}
