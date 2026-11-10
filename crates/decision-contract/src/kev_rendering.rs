//! How Kev reads a request: the text of its state, questions and options,
//! and how its answers' confidences are computed.
//!
//! Kev (`jaredpalmer/kev`) renders a JSON value as labelled text, not as
//! JSON: an object becomes `key: value` lines, a nested object or list is
//! indented under its key, a list becomes `- item` lines, and scalars are
//! written as Python's `str()` writes them, so booleans read `True` and
//! `False`. A `noul` question becomes the two options `no` and `yes`, a
//! `choice` option `name: description`, and a `score` level its rendered
//! text. Its confidences mirror TypeSafe's reference adapter: a choice's is
//! how far its top probability sits above uniform, a score's how tightly
//! the probability gathers around its most likely level.
//!
//! Each function names the reference function it mirrors in `kev/api.py` or
//! `kev/model.py`; the token sequence a model reads depends on matching it
//! character for character.

use crate::answer::{AnswerError, DecisionAnswer, answer_from_probabilities};
use crate::ordered_json::OrderedJsonValue;
use crate::python_json_text::python_float_repr;
use crate::question::{DecisionQuestion, DecisionQuestionBody};

/// The option text of a `noul` question's "false" answer, and of its "true" one.
const KEV_NOUL_OPTION_NAMES: [&str; 2] = ["no", "yes"];
/// The two characters Kev's escape puts in place of `|` in a token-like name.
const ESCAPED_TOKEN_BAR: char = '¦';

/// What one question reads as: its instructions and each option's text, in
/// the question's option order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KevQuestionText {
    pub instructions_text: String,
    pub option_texts: Vec<String>,
}

/// `render(value)`: a JSON value as the labelled text Kev reads.
#[must_use]
pub fn kev_render(value: &OrderedJsonValue) -> String {
    render_at_depth(value, 0)
}

fn render_at_depth(value: &OrderedJsonValue, indent_depth: usize) -> String {
    let indent = "  ".repeat(indent_depth);
    match value {
        OrderedJsonValue::Null => String::new(),
        OrderedJsonValue::Text(text) => text.clone(),
        OrderedJsonValue::Boolean(true) => "True".to_owned(),
        OrderedJsonValue::Boolean(false) => "False".to_owned(),
        OrderedJsonValue::SignedInteger(integer) => integer.to_string(),
        OrderedJsonValue::UnsignedInteger(integer) => integer.to_string(),
        OrderedJsonValue::Float(float) => python_float_repr(*float),
        OrderedJsonValue::Array(elements) => {
            let mut element_lines = Vec::with_capacity(elements.len());
            for element in elements {
                let element_text = render_at_depth(element, indent_depth + 1);
                element_lines.push(format!("{indent}- {}", element_text.trim_start()));
            }
            element_lines.join("\n")
        }
        OrderedJsonValue::Object(members) => {
            let mut member_lines = Vec::with_capacity(members.len());
            for object_member in members {
                member_lines.push(render_member(
                    &indent,
                    indent_depth,
                    &object_member.key,
                    &object_member.value,
                ));
            }
            member_lines.join("\n")
        }
    }
}

/// One `key: value` line, or `key:` with the nested value indented below it.
fn render_member(indent: &str, indent_depth: usize, key: &str, value: &OrderedJsonValue) -> String {
    match value {
        OrderedJsonValue::Object(_) | OrderedJsonValue::Array(_) => {
            format!(
                "{indent}{key}:\n{}",
                render_at_depth(value, indent_depth + 1)
            )
        }
        scalar_value => format!("{indent}{key}: {}", render_at_depth(scalar_value, 0)),
    }
}

/// `option_text(name, description)`: the name alone when there is no
/// description, otherwise `name: description`. Python's test is
/// `desc is None or desc == ""`, so only an empty string counts as absent.
fn option_text(option_name: &str, description: Option<&OrderedJsonValue>) -> String {
    match description {
        None | Some(OrderedJsonValue::Null) => option_name.to_owned(),
        Some(OrderedJsonValue::Text(text)) if text.is_empty() => option_name.to_owned(),
        Some(described) => format!("{option_name}: {}", kev_render(described)),
    }
}

/// `to_record`'s view of one question.
#[must_use]
pub fn kev_question_text(question: &DecisionQuestion) -> KevQuestionText {
    let instructions_text = match &question.instructions {
        Some(instructions) => kev_render(instructions),
        None => String::new(),
    };
    let option_texts = match &question.body {
        DecisionQuestionBody::Choice { options } => {
            let mut option_texts = Vec::with_capacity(options.len());
            for option in options {
                option_texts.push(option_text(&option.key, option.description.as_ref()));
            }
            option_texts
        }
        DecisionQuestionBody::Score { levels } => {
            let mut option_texts = Vec::with_capacity(levels.len());
            for level in levels {
                option_texts.push(kev_render(level));
            }
            option_texts
        }
        DecisionQuestionBody::Noul {
            false_description,
            true_description,
            ..
        } => vec![
            option_text(KEV_NOUL_OPTION_NAMES[0], false_description.as_ref()),
            option_text(KEV_NOUL_OPTION_NAMES[1], true_description.as_ref()),
        ],
    };
    KevQuestionText {
        instructions_text,
        option_texts,
    }
}

/// `user_tokens`'s escape: every `<|name|>` in caller text becomes
/// `<¦name¦>`, so no caller can write one of the control tokens Kev's
/// layout is built from. A name is one or more ASCII letters, digits or
/// underscores, as the reference's pattern `<\|([A-Za-z0-9_]+)\|>` reads it.
#[must_use]
pub fn kev_escape_control_tokens(caller_text: &str) -> String {
    let mut escaped_text = String::with_capacity(caller_text.len());
    let mut remaining_text = caller_text;
    while let Some(opening_start) = remaining_text.find("<|") {
        escaped_text.push_str(&remaining_text[..opening_start]);
        let after_opening = &remaining_text[opening_start + 2..];
        let name_length = after_opening
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if name_length > 0 && after_opening[name_length..].starts_with("|>") {
            escaped_text.push('<');
            escaped_text.push(ESCAPED_TOKEN_BAR);
            escaped_text.push_str(&after_opening[..name_length]);
            escaped_text.push(ESCAPED_TOKEN_BAR);
            escaped_text.push('>');
            remaining_text = &after_opening[name_length + 2..];
        } else {
            escaped_text.push_str("<|");
            remaining_text = after_opening;
        }
    }
    escaped_text.push_str(remaining_text);
    escaped_text
}

/// The answer Kev gives for `option_probabilities`: the shared answer, with
/// Kev's own confidences and a score legend in Kev's own text.
pub fn kev_answer_from_probabilities(
    question: &DecisionQuestion,
    option_probabilities: &[f64],
) -> Result<DecisionAnswer, AnswerError> {
    let mut answer = answer_from_probabilities(question, option_probabilities)?;
    match &mut answer {
        DecisionAnswer::Choice(choice_answer) => {
            choice_answer.distribution_confidence = kev_choice_confidence(option_probabilities);
        }
        DecisionAnswer::Score(score_answer) => {
            score_answer.distribution_confidence = kev_score_confidence(option_probabilities);
            score_answer.level_legend = kev_question_text(question).option_texts;
        }
        DecisionAnswer::Noul(_) => {}
    }
    Ok(answer)
}

/// `_normalize`: the probabilities scaled to sum to one, or uniform when
/// they sum to zero.
fn normalized(option_probabilities: &[f64]) -> Vec<f64> {
    let probability_sum: f64 = option_probabilities.iter().sum();
    let option_count = option_probabilities.len() as f64;
    if probability_sum == 0.0 {
        return vec![1.0 / option_count; option_probabilities.len()];
    }
    option_probabilities
        .iter()
        .map(|probability| probability / probability_sum)
        .collect()
}

/// `choice_confidence`: `(p_max − 1/K) / (1 − 1/K)`, 0 at uniform and 1 at
/// certainty; one option is always certain.
#[must_use]
pub fn kev_choice_confidence(option_probabilities: &[f64]) -> f64 {
    let option_count = option_probabilities.len() as f64;
    if option_probabilities.len() <= 1 {
        return 1.0;
    }
    let largest_probability = normalized(option_probabilities)
        .into_iter()
        .fold(0.0, f64::max);
    (largest_probability - 1.0 / option_count) / (1.0 - 1.0 / option_count)
}

/// `score_confidence`: `max(0, 1 − E|level − mode| / D)`, where `mode` is
/// the first most likely level and `D` the mean distance of a uniform
/// distribution from its centre.
#[must_use]
pub fn kev_score_confidence(level_probabilities: &[f64]) -> f64 {
    let level_count = level_probabilities.len();
    if level_count <= 1 {
        return 1.0;
    }
    let normalized_probabilities = normalized(level_probabilities);
    let mut mode_level = 0;
    for (level_index, level_probability) in normalized_probabilities.iter().enumerate() {
        if *level_probability > normalized_probabilities[mode_level] {
            mode_level = level_index;
        }
    }
    let centre = (level_count - 1) as f64 / 2.0;
    let mut uniform_spread = 0.0;
    let mut expected_distance = 0.0;
    for (level_index, level_probability) in normalized_probabilities.iter().enumerate() {
        uniform_spread += (level_index as f64 - centre).abs();
        expected_distance += level_probability * (level_index as f64 - mode_level as f64).abs();
    }
    uniform_spread /= level_count as f64;
    (1.0 - expected_distance / uniform_spread).max(0.0)
}

#[cfg(test)]
#[path = "kev_rendering_tests.rs"]
mod tests;
