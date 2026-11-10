//! How the Laya family words a question, its options and the state.
//!
//! Laya is a bidirectional encoder that scores one marker token per option.
//! These rules reproduce the reference implementation's text exactly, because
//! a checkpoint only answers well on the wording it was trained on: the yes/no
//! options are `false: …` then `true: …`, a score level is `level N: …`, and
//! structured values are JSON with Python's default separators.

use thiserror::Error;

use crate::criterion_text::{criterion_text, is_missing_description};
use crate::ordered_json::OrderedJsonValue;
use crate::python_json_text::{JsonTextStyle, python_json_text};
use crate::question::{DecisionQuestion, DecisionQuestionBody, DecisionQuestionKind, NoulLabels};

/// The wording a `noul` question's "false" option gets when the request gives none.
pub const DEFAULT_FALSE_DESCRIPTION: &str = "no, the statement does not hold";
/// The wording a `noul` question's "true" option gets when the request gives none.
pub const DEFAULT_TRUE_DESCRIPTION: &str = "yes, the statement holds";
const DEFAULT_FALSE_LABEL: &str = "false";
const DEFAULT_TRUE_LABEL: &str = "true";

/// Why a question cannot be rendered for Laya.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LayaRenderingError {
    #[error("question {question:?}: no 'instructions'; add the text the model should answer")]
    MissingInstructions { question: String },
}

/// The index of a question kind's row in a Laya checkpoint's type embedding.
///
/// The checkpoints were trained with this numbering, so it is part of the
/// weight format and is not the order kinds are declared in anywhere else.
#[must_use]
pub const fn laya_question_type_index(kind: DecisionQuestionKind) -> u32 {
    match kind {
        DecisionQuestionKind::Choice => 0,
        DecisionQuestionKind::Score => 1,
        DecisionQuestionKind::Noul => 2,
    }
}

/// The text that opens a question's sequence: `<type> question: <instructions>`.
///
/// Laya has no default instruction, so a question without one is rejected
/// here and never answered from its options alone.
pub fn laya_instruction_text(question: &DecisionQuestion) -> Result<String, LayaRenderingError> {
    let Some(instructions) = &question.instructions else {
        return Err(LayaRenderingError::MissingInstructions {
            question: question.name.clone(),
        });
    };
    let instructions_text = match instructions {
        OrderedJsonValue::Text(text) => text.clone(),
        structured_instructions => {
            python_json_text(structured_instructions, JsonTextStyle::PYTHON_DEFAULT)
        }
    };
    Ok(format!(
        "{} question: {instructions_text}",
        question.kind().wire_name()
    ))
}

/// One text per option, in the order Laya lays the options out.
///
/// That order is the canonical one answers are reported in: a `choice` in the
/// request's order, a `score` from level 0, and a `noul` as false then true.
#[must_use]
pub fn laya_option_texts(question: &DecisionQuestion) -> Vec<String> {
    match &question.body {
        DecisionQuestionBody::Choice { options } => {
            let mut option_texts = Vec::with_capacity(options.len());
            for option in options {
                option_texts.push(match &option.description {
                    Some(description) => {
                        format!("{}: {}", option.key, criterion_text(description))
                    }
                    None => option.key.clone(),
                });
            }
            option_texts
        }
        DecisionQuestionBody::Score { levels } => {
            let mut option_texts = Vec::with_capacity(levels.len());
            for (level_index, level) in levels.iter().enumerate() {
                option_texts.push(format!("level {level_index}: {}", criterion_text(level)));
            }
            option_texts
        }
        DecisionQuestionBody::Noul {
            false_description,
            true_description,
            labels,
        } => noul_option_texts(
            false_description.as_ref(),
            true_description.as_ref(),
            labels.as_ref(),
        ),
    }
}

fn noul_option_texts(
    false_description: Option<&OrderedJsonValue>,
    true_description: Option<&OrderedJsonValue>,
    labels: Option<&NoulLabels>,
) -> Vec<String> {
    let (false_label, true_label) = match labels {
        Some(custom_labels) => (
            custom_labels.false_label.as_str(),
            custom_labels.true_label.as_str(),
        ),
        None => (DEFAULT_FALSE_LABEL, DEFAULT_TRUE_LABEL),
    };
    vec![
        format!(
            "{false_label}: {}",
            description_or_default(false_description, DEFAULT_FALSE_DESCRIPTION)
        ),
        format!(
            "{true_label}: {}",
            description_or_default(true_description, DEFAULT_TRUE_DESCRIPTION)
        ),
    ]
}

fn description_or_default(
    description: Option<&OrderedJsonValue>,
    default_description: &str,
) -> String {
    match description {
        Some(given_description) if !is_missing_description(description) => {
            criterion_text(given_description)
        }
        _ => default_description.to_owned(),
    }
}

/// The text Laya reads a request's state as.
///
/// A string is the state itself; anything else is JSON with Python's default
/// separators and the request's own member order.
#[must_use]
pub fn laya_state_text(state: &OrderedJsonValue) -> String {
    match state {
        OrderedJsonValue::Text(text) => text.clone(),
        structured_state => python_json_text(structured_state, JsonTextStyle::PYTHON_DEFAULT),
    }
}

/// Whether a state too long for the sequence loses its beginning instead of its end.
///
/// An array is read as a conversation in time order, so the newest turn is
/// last and is the one that must survive; every other state keeps its start.
#[must_use]
pub const fn laya_truncates_state_from_the_left(state: &OrderedJsonValue) -> bool {
    matches!(state, OrderedJsonValue::Array(_))
}

#[cfg(test)]
#[path = "laya_rendering_tests.rs"]
mod tests;
