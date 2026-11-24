//! Reads the `questions` object of a System One request into typed questions.
//!
//! System One is the request shape nearly every open decision model accepts:
//! an object that maps a question name to `{type, instructions, criteria}`.
//! Parsing here, before any model is involved, means a malformed question is
//! rejected with a message that names the question and the fix, and never
//! reaches a tokenizer or the graphics processor.
//!
//! # Where this is stricter than the reference implementations
//!
//! * A label in a `choice` array must be a string. The reference also takes
//!   numbers and booleans and then reports them under string keys, so `1`,
//!   `1.0` and `true` can collide; a string is the only label that comes back
//!   as the caller wrote it.
//! * `option_order` is rejected. It changes which option each probability
//!   belongs to, and no backend here implements it.
//! * Boolean `instructions` are rejected. Numbers are accepted, as the
//!   reference accepts them, and are rendered as JSON text.
//! * A question allows at most [`MAXIMUM_OPTIONS_PER_QUESTION`] options and a
//!   request at most [`MAXIMUM_QUESTIONS_PER_REQUEST`] questions.
//!
//! The size of a state or of any text is not bounded here; the daemon bounds
//! the whole request body before it is parsed.

use thiserror::Error;

use crate::criterion_text::is_missing_description;
use crate::ordered_json::{OrderedJsonMember, OrderedJsonValue};
use crate::question::{
    ChoiceOption, DecisionQuestion, DecisionQuestionBody, DecisionQuestionKind, NoulLabels,
};

/// The most questions one request may ask.
///
/// Every question is its own sequence in a forward pass, so the count bounds
/// the batch a request can make the graphics processor allocate.
pub const MAXIMUM_QUESTIONS_PER_REQUEST: usize = 256;

/// The most options a `choice` or levels a `score` may define.
///
/// 255 is the ceiling the System One request shape documents; individual
/// model families advise far fewer and enforce their own token budgets.
pub const MAXIMUM_OPTIONS_PER_QUESTION: usize = 255;

/// Why a `questions` object cannot be answered.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum QuestionDefinitionError {
    #[error("questions must be an object mapping a question name to its definition, got {actual}")]
    QuestionsNotAnObject { actual: &'static str },
    #[error("questions must define at least one question")]
    NoQuestions,
    #[error("a request may ask at most {maximum} questions, got {actual}")]
    TooManyQuestions { actual: usize, maximum: usize },
    #[error("a question name must not be blank")]
    BlankQuestionName,
    #[error("question {question:?}: the definition must be an object, got {actual}")]
    DefinitionNotAnObject {
        question: String,
        actual: &'static str,
    },
    #[error("question {question:?}: unknown type {actual}; use choice, score or noul")]
    UnknownType { question: String, actual: String },
    #[error("question {question:?}: {reason}")]
    InvalidInstructions {
        question: String,
        reason: &'static str,
    },
    #[error("question {question:?}: {reason}")]
    InvalidCriteria { question: String, reason: String },
    #[error("question {question:?}: at most {maximum} options are allowed, got {actual}")]
    TooManyOptions {
        question: String,
        actual: usize,
        maximum: usize,
    },
    #[error(
        "question {question:?}: choice label {label:?} is repeated; labels are the answer keys, so each option needs its own"
    )]
    RepeatedChoiceLabel { question: String, label: String },
    #[error("question {question:?}: {reason}")]
    InvalidLabels {
        question: String,
        reason: &'static str,
    },
    #[error("question {question:?}: the field {field:?} is not supported")]
    UnsupportedField {
        question: String,
        field: &'static str,
    },
}

/// Parses and validates every question of a System One `questions` object.
///
/// Questions are returned in the order the request listed them, which is the
/// order their answers are reported in.
pub fn parse_system_one_questions(
    questions: &OrderedJsonValue,
) -> Result<Vec<DecisionQuestion>, QuestionDefinitionError> {
    let Some(question_members) = questions.as_object() else {
        return Err(QuestionDefinitionError::QuestionsNotAnObject {
            actual: questions.type_name(),
        });
    };
    if question_members.is_empty() {
        return Err(QuestionDefinitionError::NoQuestions);
    }
    if question_members.len() > MAXIMUM_QUESTIONS_PER_REQUEST {
        return Err(QuestionDefinitionError::TooManyQuestions {
            actual: question_members.len(),
            maximum: MAXIMUM_QUESTIONS_PER_REQUEST,
        });
    }
    let mut parsed_questions = Vec::with_capacity(question_members.len());
    for question_member in question_members {
        parsed_questions.push(parse_question(question_member)?);
    }
    Ok(parsed_questions)
}

fn parse_question(
    question_member: &OrderedJsonMember,
) -> Result<DecisionQuestion, QuestionDefinitionError> {
    let question_name = &question_member.key;
    if question_name.trim().is_empty() {
        return Err(QuestionDefinitionError::BlankQuestionName);
    }
    let definition = &question_member.value;
    if definition.as_object().is_none() {
        return Err(QuestionDefinitionError::DefinitionNotAnObject {
            question: question_name.clone(),
            actual: definition.type_name(),
        });
    }
    let kind = parse_kind(question_name, definition)?;
    reject_unsupported_fields(question_name, definition, kind)?;
    let instructions = parse_instructions(question_name, definition.member("instructions"))?;
    let criteria = definition.member("criteria");
    let body = match kind {
        DecisionQuestionKind::Choice => parse_choice_body(question_name, criteria)?,
        DecisionQuestionKind::Score => parse_score_body(question_name, criteria)?,
        DecisionQuestionKind::Noul => {
            parse_noul_body(question_name, criteria, definition.member("labels"))?
        }
    };
    Ok(DecisionQuestion {
        name: question_name.clone(),
        instructions,
        body,
    })
}

fn parse_kind(
    question_name: &str,
    definition: &OrderedJsonValue,
) -> Result<DecisionQuestionKind, QuestionDefinitionError> {
    let type_field = definition.member("type");
    let known_kind = type_field
        .and_then(OrderedJsonValue::as_text)
        .and_then(DecisionQuestionKind::from_wire_name);
    match known_kind {
        Some(kind) => Ok(kind),
        None => Err(QuestionDefinitionError::UnknownType {
            question: question_name.to_owned(),
            actual: describe_type_field(type_field),
        }),
    }
}

fn describe_type_field(type_field: Option<&OrderedJsonValue>) -> String {
    match type_field {
        None => "(missing)".to_owned(),
        Some(OrderedJsonValue::Text(text)) => format!("{text:?}"),
        Some(other_value) => format!("of JSON type {}", other_value.type_name()),
    }
}

/// Rejects fields whose meaning this implementation does not carry out.
///
/// Ignoring them would answer a different question than the caller asked:
/// `option_order` changes which option each probability belongs to, and
/// `labels` on a `choice` or `score` has no defined meaning.
fn reject_unsupported_fields(
    question_name: &str,
    definition: &OrderedJsonValue,
    kind: DecisionQuestionKind,
) -> Result<(), QuestionDefinitionError> {
    if definition.member("option_order").is_some() {
        return Err(QuestionDefinitionError::UnsupportedField {
            question: question_name.to_owned(),
            field: "option_order",
        });
    }
    if kind != DecisionQuestionKind::Noul && definition.member("labels").is_some() {
        return Err(QuestionDefinitionError::InvalidLabels {
            question: question_name.to_owned(),
            reason: "'labels' is only supported for noul questions",
        });
    }
    Ok(())
}

fn parse_instructions(
    question_name: &str,
    instructions: Option<&OrderedJsonValue>,
) -> Result<Option<OrderedJsonValue>, QuestionDefinitionError> {
    let rejection_reason = match instructions {
        None | Some(OrderedJsonValue::Null) => return Ok(None),
        Some(OrderedJsonValue::Text(text)) if text.trim().is_empty() => {
            "'instructions' must not be empty; add the text the model should answer"
        }
        Some(OrderedJsonValue::Array(elements)) if elements.is_empty() => {
            "'instructions' must not be empty; add the text the model should answer"
        }
        Some(OrderedJsonValue::Object(members)) if members.is_empty() => {
            "'instructions' must not be empty; add the text the model should answer"
        }
        Some(OrderedJsonValue::Boolean(_)) => {
            "'instructions' must be a string, object, array or number"
        }
        Some(usable_instructions) => return Ok(Some(usable_instructions.clone())),
    };
    Err(QuestionDefinitionError::InvalidInstructions {
        question: question_name.to_owned(),
        reason: rejection_reason,
    })
}

fn parse_choice_body(
    question_name: &str,
    criteria: Option<&OrderedJsonValue>,
) -> Result<DecisionQuestionBody, QuestionDefinitionError> {
    let options = match criteria {
        Some(OrderedJsonValue::Object(members)) => choice_options_from_object(members),
        Some(OrderedJsonValue::Array(labels)) => choice_options_from_labels(question_name, labels)?,
        _ => {
            return Err(invalid_criteria(
                question_name,
                "a choice question takes 'criteria' as an object of label -> description, or an array of labels",
            ));
        }
    };
    if options.is_empty() {
        return Err(invalid_criteria(
            question_name,
            "a choice question needs at least one criterion",
        ));
    }
    check_option_count(question_name, options.len())?;
    Ok(DecisionQuestionBody::Choice { options })
}

fn choice_options_from_object(members: &[OrderedJsonMember]) -> Vec<ChoiceOption> {
    let mut options = Vec::with_capacity(members.len());
    for criterion_member in members {
        let description = if is_missing_description(Some(&criterion_member.value)) {
            None
        } else {
            Some(criterion_member.value.clone())
        };
        options.push(ChoiceOption {
            key: criterion_member.key.clone(),
            description,
        });
    }
    options
}

fn choice_options_from_labels(
    question_name: &str,
    labels: &[OrderedJsonValue],
) -> Result<Vec<ChoiceOption>, QuestionDefinitionError> {
    let mut options: Vec<ChoiceOption> = Vec::with_capacity(labels.len());
    for label in labels {
        let Some(label_text) = label.as_text() else {
            return Err(invalid_criteria(
                question_name,
                "every label in a choice 'criteria' array must be a string",
            ));
        };
        for earlier_option in &options {
            if earlier_option.key == label_text {
                return Err(QuestionDefinitionError::RepeatedChoiceLabel {
                    question: question_name.to_owned(),
                    label: label_text.to_owned(),
                });
            }
        }
        options.push(ChoiceOption {
            key: label_text.to_owned(),
            description: None,
        });
    }
    Ok(options)
}

fn parse_score_body(
    question_name: &str,
    criteria: Option<&OrderedJsonValue>,
) -> Result<DecisionQuestionBody, QuestionDefinitionError> {
    let Some(OrderedJsonValue::Array(levels)) = criteria else {
        return Err(invalid_criteria(
            question_name,
            "a score question takes 'criteria' as an array of level descriptions, index 0 first",
        ));
    };
    if levels.is_empty() {
        return Err(invalid_criteria(
            question_name,
            "a score question needs at least one level",
        ));
    }
    for level in levels {
        if *level == OrderedJsonValue::Null {
            return Err(invalid_criteria(
                question_name,
                "a score level is null; give every level a description, index 0 first",
            ));
        }
    }
    check_option_count(question_name, levels.len())?;
    Ok(DecisionQuestionBody::Score {
        levels: levels.clone(),
    })
}

fn parse_noul_body(
    question_name: &str,
    criteria: Option<&OrderedJsonValue>,
    labels: Option<&OrderedJsonValue>,
) -> Result<DecisionQuestionBody, QuestionDefinitionError> {
    let (false_description, true_description) = match criteria {
        None | Some(OrderedJsonValue::Null) => (None, None),
        Some(OrderedJsonValue::Object(members)) => noul_descriptions(question_name, members)?,
        Some(_) => {
            return Err(invalid_criteria(
                question_name,
                "a noul question takes 'criteria' as an object with optional 'true' and 'false' descriptions, or omits it",
            ));
        }
    };
    Ok(DecisionQuestionBody::Noul {
        false_description,
        true_description,
        labels: parse_noul_labels(question_name, labels)?,
    })
}

/// The `(false, true)` descriptions of a `noul` criteria object.
///
/// Keys are matched without regard to case, as the reference does, and a key
/// written twice in different cases keeps the later description.
fn noul_descriptions(
    question_name: &str,
    members: &[OrderedJsonMember],
) -> Result<(Option<OrderedJsonValue>, Option<OrderedJsonValue>), QuestionDefinitionError> {
    let mut false_description = None;
    let mut true_description = None;
    for criterion_member in members {
        let description = usable_description(&criterion_member.value);
        match criterion_member.key.to_lowercase().as_str() {
            "false" => false_description = description,
            "true" => true_description = description,
            _ => {
                return Err(invalid_criteria(
                    question_name,
                    "a noul question takes 'criteria' keyed only 'true' and 'false'; to word the answer differently, set 'labels'",
                ));
            }
        }
    }
    Ok((false_description, true_description))
}

fn usable_description(description: &OrderedJsonValue) -> Option<OrderedJsonValue> {
    if is_missing_description(Some(description)) {
        None
    } else {
        Some(description.clone())
    }
}

const NOUL_LABELS_RULE: &str =
    "noul 'labels' must map exactly 'false' and 'true' to distinct non-empty strings";

fn parse_noul_labels(
    question_name: &str,
    labels: Option<&OrderedJsonValue>,
) -> Result<Option<NoulLabels>, QuestionDefinitionError> {
    let Some(labels) = labels else {
        return Ok(None);
    };
    let invalid_labels = QuestionDefinitionError::InvalidLabels {
        question: question_name.to_owned(),
        reason: NOUL_LABELS_RULE,
    };
    let Some(label_members) = labels.as_object() else {
        return Err(invalid_labels);
    };
    let false_label = labels.member("false").and_then(OrderedJsonValue::as_text);
    let true_label = labels.member("true").and_then(OrderedJsonValue::as_text);
    let (Some(false_label), Some(true_label)) = (false_label, true_label) else {
        return Err(invalid_labels);
    };
    let false_label = false_label.trim();
    let true_label = true_label.trim();
    let has_only_the_two_keys = label_members.len() == 2;
    if !has_only_the_two_keys
        || false_label.is_empty()
        || true_label.is_empty()
        || false_label == true_label
    {
        return Err(invalid_labels);
    }
    Ok(Some(NoulLabels {
        false_label: false_label.to_owned(),
        true_label: true_label.to_owned(),
    }))
}

fn check_option_count(
    question_name: &str,
    option_count: usize,
) -> Result<(), QuestionDefinitionError> {
    if option_count > MAXIMUM_OPTIONS_PER_QUESTION {
        return Err(QuestionDefinitionError::TooManyOptions {
            question: question_name.to_owned(),
            actual: option_count,
            maximum: MAXIMUM_OPTIONS_PER_QUESTION,
        });
    }
    Ok(())
}

fn invalid_criteria(question_name: &str, reason: &str) -> QuestionDefinitionError {
    QuestionDefinitionError::InvalidCriteria {
        question: question_name.to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
#[path = "system_one_questions_tests.rs"]
mod tests;
