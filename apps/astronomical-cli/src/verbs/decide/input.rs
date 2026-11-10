//! Turns a `decide` invocation into the state and questions the daemon
//! receives, before any daemon is contacted.
//!
//! Reading and checking the input is kept apart from the daemon exchange so
//! a malformed invocation is reported as the usage mistake it is, whether or
//! not a daemon is running.

use std::io::Read;

use astronomical_decision_contract::{OrderedJsonMember, OrderedJsonValue};

use super::{DecideArguments, DecideError, DecideQuestionsSource};

/// The name a question asked with `--ask` is answered under.
pub const ASK_QUESTION_NAME: &str = "answer";

/// The questions as the System One object the daemon expects.
pub(super) fn resolve_questions(
    questions_source: &DecideQuestionsSource,
) -> Result<OrderedJsonValue, DecideError> {
    let questions_json_text = match questions_source {
        DecideQuestionsSource::Ask(question_text) => {
            if question_text.trim().is_empty() {
                return Err(DecideError::EmptyQuestion);
            }
            return Ok(single_yes_or_no_question(question_text));
        }
        DecideQuestionsSource::InlineJson(questions_json_text) => questions_json_text.clone(),
        DecideQuestionsSource::File(file_path) => match std::fs::read_to_string(file_path) {
            Ok(file_text) => file_text,
            Err(read_error) => {
                return Err(DecideError::InputFileUnreadable {
                    file_path: file_path.clone(),
                    cause: read_error.to_string(),
                });
            }
        },
    };
    match serde_json::from_str::<OrderedJsonValue>(&questions_json_text) {
        Ok(questions) => Ok(questions),
        Err(json_error) => Err(DecideError::QuestionsNotJson {
            cause: json_error.to_string(),
        }),
    }
}

/// `{"answer": {"type": "noul", "instructions": <question>}}`.
fn single_yes_or_no_question(question_text: &str) -> OrderedJsonValue {
    let question_definition = OrderedJsonValue::Object(vec![
        OrderedJsonMember {
            key: "type".to_owned(),
            value: OrderedJsonValue::Text("noul".to_owned()),
        },
        OrderedJsonMember {
            key: "instructions".to_owned(),
            value: OrderedJsonValue::Text(question_text.to_owned()),
        },
    ]);
    OrderedJsonValue::Object(vec![OrderedJsonMember {
        key: ASK_QUESTION_NAME.to_owned(),
        value: question_definition,
    }])
}

/// State text > file contents > stdin read to EOF; parsed as JSON only when
/// the caller said it is JSON. An empty state is a usage failure.
pub(super) fn resolve_state(
    decide_arguments: &DecideArguments,
    stdin: &mut dyn Read,
) -> Result<OrderedJsonValue, DecideError> {
    let state_text = if let Some(state_text) = &decide_arguments.state_text {
        state_text.clone()
    } else if let Some(file_path) = &decide_arguments.state_file_path {
        match std::fs::read_to_string(file_path) {
            Ok(file_text) => file_text,
            Err(read_error) => {
                return Err(DecideError::InputFileUnreadable {
                    file_path: file_path.clone(),
                    cause: read_error.to_string(),
                });
            }
        }
    } else {
        let mut stdin_text = String::new();
        if let Err(read_error) = stdin.read_to_string(&mut stdin_text) {
            return Err(DecideError::StdinUnreadable {
                cause: read_error.to_string(),
            });
        }
        stdin_text
    };
    if state_text.trim().is_empty() {
        return Err(DecideError::StateRequired);
    }
    if !decide_arguments.state_is_json {
        return Ok(OrderedJsonValue::Text(state_text));
    }
    match serde_json::from_str::<OrderedJsonValue>(&state_text) {
        Ok(state) => Ok(state),
        Err(json_error) => Err(DecideError::StateNotJson {
            cause: json_error.to_string(),
        }),
    }
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod tests;
