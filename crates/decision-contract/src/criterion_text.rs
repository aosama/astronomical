//! The text a criterion, level or instruction is shown to a model as.

use crate::ordered_json::OrderedJsonValue;
use crate::python_json_text::{JsonTextStyle, python_json_text};

/// Renders one criterion value as text.
///
/// A string is used as written. Anything structured becomes single-line JSON
/// in the reference implementations' default style, so a rubric written as an
/// object reads as JSON to the model and comes back the same way in a `score`
/// answer's legend.
#[must_use]
pub fn criterion_text(criterion: &OrderedJsonValue) -> String {
    match criterion {
        OrderedJsonValue::Text(text) => text.clone(),
        structured_criterion => {
            python_json_text(structured_criterion, JsonTextStyle::PYTHON_DEFAULT)
        }
    }
}

/// Whether a description says nothing: absent, JSON `null`, or the empty string.
///
/// `0` and `false` are real descriptions, so only these three count as missing.
#[must_use]
pub fn is_missing_description(description: Option<&OrderedJsonValue>) -> bool {
    match description {
        None | Some(OrderedJsonValue::Null) => true,
        Some(OrderedJsonValue::Text(text)) => text.is_empty(),
        Some(_) => false,
    }
}

#[cfg(test)]
#[path = "criterion_text_tests.rs"]
mod tests;
