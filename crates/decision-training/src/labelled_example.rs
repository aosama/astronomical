//! One labelled example: a state, the questions asked about it, and the
//! answer that is right for each question.
//!
//! An example is one JSON object per line, in the shape `astronomical decide`
//! already takes plus the labels:
//!
//! ```json
//! {"state": "...", "questions": {"refund": {"type": "noul", "instructions": "..."}},
//!  "labels": {"refund": true}}
//! ```
//!
//! A `choice` label is the key of the right option, a `score` label is the
//! index of the right level (0 is the lowest), and a `noul` label is `true` or
//! `false`. Every question must be labelled and every label must name a
//! question: a silently unlabelled question would shrink the evidence without
//! anyone noticing, and a label for a misspelled question name would be lost.

use astronomical_decision_contract::question::{NOUL_FALSE_INDEX, NOUL_TRUE_INDEX};
use astronomical_decision_contract::{
    DecisionQuestion, DecisionQuestionBody, OrderedJsonValue, QuestionDefinitionError,
    parse_system_one_questions,
};
use thiserror::Error;

/// One validated line of a labelled examples file.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelledExample {
    /// The 1-based line the example came from, for messages.
    pub line_number: usize,
    /// The state, sent to the model exactly as written.
    pub state: OrderedJsonValue,
    /// The `questions` object, sent to the model exactly as written.
    pub questions_document: OrderedJsonValue,
    /// Every question with the index of its right answer.
    pub labelled_questions: Vec<LabelledQuestion>,
}

/// One question and the canonical index of its right answer.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelledQuestion {
    pub question: DecisionQuestion,
    /// In the contract's canonical option order: a `choice` in the order the
    /// question lists its options, a `score` from level 0, a `noul` as false
    /// then true. Model outcomes report logits in this same order.
    pub correct_option_index: usize,
}

/// Why one line cannot be used as a labelled example.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LabelledExampleError {
    #[error("line {line_number} is not JSON: {cause}")]
    NotJson { line_number: usize, cause: String },
    #[error("line {line_number} must be an object with state, questions and labels")]
    NotAnObject { line_number: usize },
    #[error("line {line_number} has no `{member_name}`")]
    MissingMember {
        line_number: usize,
        member_name: &'static str,
    },
    #[error("line {line_number}: {source}")]
    InvalidQuestions {
        line_number: usize,
        source: QuestionDefinitionError,
    },
    #[error(
        "line {line_number}: `labels` must be an object mapping a question name to its right answer"
    )]
    LabelsNotAnObject { line_number: usize },
    #[error("line {line_number}: question {question:?} has no label")]
    UnlabelledQuestion {
        line_number: usize,
        question: String,
    },
    #[error("line {line_number}: there is a label for {question:?}, but no question has that name")]
    LabelForUnknownQuestion {
        line_number: usize,
        question: String,
    },
    #[error("line {line_number}: the label for {question:?} {problem}")]
    UnusableLabel {
        line_number: usize,
        question: String,
        problem: String,
    },
    #[error("the examples file holds no examples")]
    NoExamples,
}

/// Every example in a labelled examples file; blank lines are skipped.
pub fn parse_labelled_examples(
    examples_text: &str,
) -> Result<Vec<LabelledExample>, LabelledExampleError> {
    let mut labelled_examples = Vec::new();
    for (line_index, line_text) in examples_text.lines().enumerate() {
        if line_text.trim().is_empty() {
            continue;
        }
        labelled_examples.push(parse_labelled_example(line_text, line_index + 1)?);
    }
    if labelled_examples.is_empty() {
        return Err(LabelledExampleError::NoExamples);
    }
    Ok(labelled_examples)
}

/// One example from one line of text.
pub fn parse_labelled_example(
    line_text: &str,
    line_number: usize,
) -> Result<LabelledExample, LabelledExampleError> {
    let line_document = match serde_json::from_str::<OrderedJsonValue>(line_text) {
        Ok(line_document) => line_document,
        Err(json_error) => {
            return Err(LabelledExampleError::NotJson {
                line_number,
                cause: json_error.to_string(),
            });
        }
    };
    if line_document.as_object().is_none() {
        return Err(LabelledExampleError::NotAnObject { line_number });
    }
    let state = required_member(&line_document, "state", line_number)?;
    let questions_document = required_member(&line_document, "questions", line_number)?;
    let labels = required_member(&line_document, "labels", line_number)?;
    let questions = match parse_system_one_questions(questions_document) {
        Ok(questions) => questions,
        Err(source) => {
            return Err(LabelledExampleError::InvalidQuestions {
                line_number,
                source,
            });
        }
    };
    let labelled_questions = label_questions(questions, labels, line_number)?;
    Ok(LabelledExample {
        line_number,
        state: state.clone(),
        questions_document: questions_document.clone(),
        labelled_questions,
    })
}

fn required_member<'document>(
    line_document: &'document OrderedJsonValue,
    member_name: &'static str,
    line_number: usize,
) -> Result<&'document OrderedJsonValue, LabelledExampleError> {
    line_document
        .member(member_name)
        .ok_or(LabelledExampleError::MissingMember {
            line_number,
            member_name,
        })
}

fn label_questions(
    questions: Vec<DecisionQuestion>,
    labels: &OrderedJsonValue,
    line_number: usize,
) -> Result<Vec<LabelledQuestion>, LabelledExampleError> {
    let Some(label_members) = labels.as_object() else {
        return Err(LabelledExampleError::LabelsNotAnObject { line_number });
    };
    for label_member in label_members {
        if !has_question_named(&questions, &label_member.key) {
            return Err(LabelledExampleError::LabelForUnknownQuestion {
                line_number,
                question: label_member.key.clone(),
            });
        }
    }
    let mut labelled_questions = Vec::with_capacity(questions.len());
    for question in questions {
        let Some(label) = labels.member(&question.name) else {
            return Err(LabelledExampleError::UnlabelledQuestion {
                line_number,
                question: question.name,
            });
        };
        let correct_option_index = match correct_option_index(&question, label) {
            Ok(correct_option_index) => correct_option_index,
            Err(problem) => {
                return Err(LabelledExampleError::UnusableLabel {
                    line_number,
                    question: question.name,
                    problem,
                });
            }
        };
        labelled_questions.push(LabelledQuestion {
            question,
            correct_option_index,
        });
    }
    Ok(labelled_questions)
}

fn has_question_named(questions: &[DecisionQuestion], question_name: &str) -> bool {
    for question in questions {
        if question.name == question_name {
            return true;
        }
    }
    false
}

/// The canonical index a label names, or what is wrong with the label.
fn correct_option_index(
    question: &DecisionQuestion,
    label: &OrderedJsonValue,
) -> Result<usize, String> {
    match &question.body {
        DecisionQuestionBody::Choice { options } => {
            let Some(label_key) = label.as_text() else {
                return Err("must be the key of one of its options, as text".to_owned());
            };
            for (option_index, option) in options.iter().enumerate() {
                if option.key == label_key {
                    return Ok(option_index);
                }
            }
            Err(format!("is {label_key:?}, which is not one of its options"))
        }
        DecisionQuestionBody::Score { levels } => match label {
            OrderedJsonValue::SignedInteger(level_index)
                if *level_index >= 0 && (*level_index as u64) < levels.len() as u64 =>
            {
                Ok(*level_index as usize)
            }
            _ => Err(format!(
                "must be a level index from 0 to {}",
                levels.len().saturating_sub(1)
            )),
        },
        DecisionQuestionBody::Noul { .. } => match label {
            OrderedJsonValue::Boolean(true) => Ok(NOUL_TRUE_INDEX),
            OrderedJsonValue::Boolean(false) => Ok(NOUL_FALSE_INDEX),
            _ => Err("must be true or false".to_owned()),
        },
    }
}

#[cfg(test)]
#[path = "labelled_example_tests.rs"]
mod tests;
