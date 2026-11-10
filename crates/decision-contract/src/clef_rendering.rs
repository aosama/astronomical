//! How Clef reads a request: the text of its state, questions and options,
//! and the order it scores the options in.
//!
//! Clef (`Cloudflare/clef-flash`) writes a string as it is and any other
//! JSON value as `json.dumps(value, ensure_ascii=False, separators=(",",
//! ":"), sort_keys=True)`. Each option is the JSON object
//! `{"description": …, "option_id": …}` (the description left out when
//! there is none). A `choice` question's options are scored in key order,
//! not the caller's; a `noul` question's are `true` then `false`, each with
//! a default description the caller's criteria may replace; a `score`
//! question's are its levels, `"0"` upward.
//!
//! Each function names the reference function it mirrors in
//! `joint_schema_model.py`; the token sequence the model reads depends on
//! matching it character for character.

use crate::ordered_json::{OrderedJsonMember, OrderedJsonValue};
use crate::python_json_text::{JsonTextStyle, python_json_text};
use crate::question::{DecisionQuestion, DecisionQuestionBody, NOUL_FALSE_INDEX};

/// `QUESTION_TYPES`: the type identifier the head embeds for each kind.
const CLEF_NOUL_TYPE_ID: u32 = 0;
const CLEF_CHOICE_TYPE_ID: u32 = 1;
const CLEF_SCORE_TYPE_ID: u32 = 2;
/// `question_options`' defaults for a `noul` question without criteria.
const CLEF_DEFAULT_TRUE_DESCRIPTION: &str = "The proposition is true or the answer is yes.";
const CLEF_DEFAULT_FALSE_DESCRIPTION: &str = "The proposition is false or the answer is no.";

/// One option as Clef reads it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClefOptionText {
    /// The identifier Clef reports the option's probability under.
    pub option_id: String,
    /// The option's text: its identifier and description as JSON.
    pub semantics_text: String,
    /// Where this option sits in the contract's canonical order (the
    /// caller's order for a choice, `[false, true]` for a noul, the level
    /// for a score), so probabilities can be put back in that order.
    pub canonical_index: usize,
}

/// One question as Clef reads it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClefQuestionText {
    /// `QUESTION_TYPES[type]`.
    pub type_id: u32,
    /// The instructions, or the question's name when it gave none.
    pub instructions_text: String,
    /// The options in the order Clef scores them.
    pub options: Vec<ClefOptionText>,
}

/// `render`: a string as written, anything else as compact sorted JSON.
#[must_use]
pub fn clef_render(value: &OrderedJsonValue) -> String {
    match value {
        OrderedJsonValue::Text(text) => text.clone(),
        other_value => python_json_text(other_value, JsonTextStyle::COMPACT_SORTED_KEYS),
    }
}

/// `encode_record`'s view of one question.
#[must_use]
pub fn clef_question_text(question: &DecisionQuestion) -> ClefQuestionText {
    let instructions_text = match &question.instructions {
        None => question.name.clone(),
        Some(OrderedJsonValue::Text(text)) if text.is_empty() => question.name.clone(),
        Some(instructions) => clef_render(instructions),
    };
    let (type_id, options) = match &question.body {
        DecisionQuestionBody::Choice { options } => {
            let mut keyed_options = Vec::with_capacity(options.len());
            for (canonical_index, option) in options.iter().enumerate() {
                keyed_options.push(clef_option(
                    &option.key,
                    option.description.as_ref(),
                    canonical_index,
                ));
            }
            // `sorted(criteria.items())`: keys are unique, so this is key
            // order, and Rust's byte order on UTF-8 is Python's code-point order.
            keyed_options.sort_by(|left, right| left.option_id.cmp(&right.option_id));
            (CLEF_CHOICE_TYPE_ID, keyed_options)
        }
        DecisionQuestionBody::Score { levels } => {
            let mut level_options = Vec::with_capacity(levels.len());
            for (level_index, level) in levels.iter().enumerate() {
                level_options.push(clef_option(
                    &level_index.to_string(),
                    Some(level),
                    level_index,
                ));
            }
            (CLEF_SCORE_TYPE_ID, level_options)
        }
        DecisionQuestionBody::Noul {
            false_description,
            true_description,
            ..
        } => (
            CLEF_NOUL_TYPE_ID,
            noul_options(true_description.as_ref(), false_description.as_ref()),
        ),
    };
    ClefQuestionText {
        type_id,
        instructions_text,
        options,
    }
}

/// `true` then `false`, each with the caller's description or Clef's default.
fn noul_options(
    true_description: Option<&OrderedJsonValue>,
    false_description: Option<&OrderedJsonValue>,
) -> Vec<ClefOptionText> {
    let default_true = OrderedJsonValue::Text(CLEF_DEFAULT_TRUE_DESCRIPTION.to_owned());
    let default_false = OrderedJsonValue::Text(CLEF_DEFAULT_FALSE_DESCRIPTION.to_owned());
    vec![
        clef_option(
            "true",
            Some(true_description.unwrap_or(&default_true)),
            1 - NOUL_FALSE_INDEX,
        ),
        clef_option(
            "false",
            Some(false_description.unwrap_or(&default_false)),
            NOUL_FALSE_INDEX,
        ),
    ]
}

/// One option's text: `render({"option_id": id, "description": d})`, the
/// description left out when it is null.
fn clef_option(
    option_id: &str,
    description: Option<&OrderedJsonValue>,
    canonical_index: usize,
) -> ClefOptionText {
    let mut semantics_members = vec![OrderedJsonMember {
        key: "option_id".to_owned(),
        value: OrderedJsonValue::Text(option_id.to_owned()),
    }];
    if let Some(described) = description
        && *described != OrderedJsonValue::Null
    {
        semantics_members.push(OrderedJsonMember {
            key: "description".to_owned(),
            value: described.clone(),
        });
    }
    ClefOptionText {
        option_id: option_id.to_owned(),
        semantics_text: clef_render(&OrderedJsonValue::Object(semantics_members)),
        canonical_index,
    }
}

#[cfg(test)]
#[path = "clef_rendering_tests.rs"]
mod tests;
