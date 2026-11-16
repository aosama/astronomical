//! The built-in check set: questions nobody labels, whose shipped answers
//! training must keep.
//!
//! Training changes layers every question passes through, so a run
//! that fits a person's questions could quietly change answers to questions
//! they never asked about. These 24 Romeo and Juliet lines, each asked five
//! questions covering every question kind and 2, 3, 4, 7 and 12 options, are
//! scored with the shipped model before training; training is pulled toward
//! keeping those answers, and the gate refuses a head that changes more than
//! one in twenty of them.

use astronomical_decision_contract::OrderedJsonValue;

/// One check-set line: a state and its unlabelled questions.
pub(super) struct CheckExample {
    pub(super) state: OrderedJsonValue,
    pub(super) questions_document: OrderedJsonValue,
}

const CHECK_SET_TEXT: &str = include_str!("check_set_romeo_and_juliet.jsonl");

/// Every check-set line, or why the fixture could not be read (a build defect).
pub(super) fn built_in_check_examples() -> Result<Vec<CheckExample>, String> {
    let mut check_examples = Vec::new();
    for (line_index, line_text) in CHECK_SET_TEXT.lines().enumerate() {
        if line_text.trim().is_empty() {
            continue;
        }
        let line_document = match serde_json::from_str::<OrderedJsonValue>(line_text) {
            Ok(line_document) => line_document,
            Err(json_error) => {
                return Err(format!("check set line {}: {json_error}", line_index + 1));
            }
        };
        let (Some(state), Some(questions_document)) = (
            line_document.member("state"),
            line_document.member("questions"),
        ) else {
            return Err(format!(
                "check set line {} lacks a state or questions",
                line_index + 1
            ));
        };
        check_examples.push(CheckExample {
            state: state.clone(),
            questions_document: questions_document.clone(),
        });
    }
    Ok(check_examples)
}

#[cfg(test)]
#[path = "check_set_tests.rs"]
mod tests;
