//! Writes answers in the System One response shape.
//!
//! System One is the response shape existing decision-model clients read:
//! an `answers` object keyed by question name, where a `choice` carries its
//! chosen key and a probability per option, a `score` its expected level and
//! a probability per level, and a `noul` the probability that the statement
//! holds. The REST endpoint and the command-line tool both print this shape,
//! so it is built once here.
//!
//! The response is built as ordered JSON so answers come back in the order
//! the questions were asked, and probabilities in the order the options were
//! listed.

use crate::answer::DecisionAnswer;
use crate::ordered_json::{OrderedJsonMember, OrderedJsonValue};
use crate::outcome::QuestionOutcome;

/// Decimal places every probability, score and confidence is rounded to,
/// matching the reference implementations' responses.
const REPORTED_DECIMAL_PLACES_SCALE: f64 = 10_000.0;

/// The whole System One response body for one answered request.
#[must_use]
pub fn system_one_response(
    model_id: &str,
    question_outcomes: &[QuestionOutcome],
    input_token_count: u32,
) -> OrderedJsonValue {
    let mut answer_members = Vec::with_capacity(question_outcomes.len());
    for question_outcome in question_outcomes {
        answer_members.push(member(
            &question_outcome.question_name,
            system_one_answer(question_outcome),
        ));
    }
    OrderedJsonValue::Object(vec![
        member("model", OrderedJsonValue::Text(model_id.to_owned())),
        member("answers", OrderedJsonValue::Object(answer_members)),
        member(
            "usage",
            OrderedJsonValue::Object(vec![
                member(
                    "input_tokens",
                    OrderedJsonValue::SignedInteger(i64::from(input_token_count)),
                ),
                // A decision model generates nothing.
                member("output_tokens", OrderedJsonValue::SignedInteger(0)),
            ]),
        ),
    ])
}

/// One question's answer in the System One shape, with how much of the input
/// the model read under `input`.
#[must_use]
pub fn system_one_answer(question_outcome: &QuestionOutcome) -> OrderedJsonValue {
    let mut answer_members = match &question_outcome.answer {
        DecisionAnswer::Choice(choice_answer) => {
            let mut probability_members =
                Vec::with_capacity(choice_answer.option_probabilities.len());
            for option_probability in &choice_answer.option_probabilities {
                probability_members.push(member(
                    &option_probability.key,
                    reported_number(option_probability.probability),
                ));
            }
            vec![
                member("type", OrderedJsonValue::Text("choice".to_owned())),
                member(
                    "choice",
                    OrderedJsonValue::Text(choice_answer.chosen_key.clone()),
                ),
                member(
                    "probabilities",
                    OrderedJsonValue::Object(probability_members),
                ),
                member(
                    "confidence",
                    reported_number(choice_answer.distribution_confidence),
                ),
                member(
                    "answer_confidence",
                    reported_number(choice_answer.answer_confidence),
                ),
            ]
        }
        DecisionAnswer::Score(score_answer) => {
            let mut legend_members = Vec::with_capacity(score_answer.level_legend.len());
            let mut probability_members = Vec::with_capacity(score_answer.level_legend.len());
            for (level_index, level_text) in score_answer.level_legend.iter().enumerate() {
                legend_members.push(member(
                    &level_index.to_string(),
                    OrderedJsonValue::Text(level_text.clone()),
                ));
            }
            for (level_index, level_probability) in
                score_answer.level_probabilities.iter().enumerate()
            {
                probability_members.push(member(
                    &level_index.to_string(),
                    reported_number(*level_probability),
                ));
            }
            vec![
                member("type", OrderedJsonValue::Text("score".to_owned())),
                member("score", reported_number(score_answer.expected_level)),
                member("legend", OrderedJsonValue::Object(legend_members)),
                member(
                    "probabilities",
                    OrderedJsonValue::Object(probability_members),
                ),
                member(
                    "confidence",
                    reported_number(score_answer.distribution_confidence),
                ),
                member(
                    "answer_confidence",
                    reported_number(score_answer.answer_confidence),
                ),
            ]
        }
        DecisionAnswer::Noul(noul_answer) => vec![
            member("type", OrderedJsonValue::Text("noul".to_owned())),
            member("noul", reported_number(noul_answer.probability_true)),
            member("confidence", reported_number(noul_answer.answer_confidence)),
            member(
                "answer_confidence",
                reported_number(noul_answer.answer_confidence),
            ),
        ],
    };
    answer_members.push(member("input", input_coverage(question_outcome)));
    OrderedJsonValue::Object(answer_members)
}

/// What the model read for one question, so a caller can see an answer was
/// made from a cut state or from options the cut made indistinguishable.
fn input_coverage(question_outcome: &QuestionOutcome) -> OrderedJsonValue {
    let coverage = question_outcome.input_coverage;
    let option_token_cap = match coverage.option_token_cap {
        Some(token_cap) => OrderedJsonValue::SignedInteger(i64::from(token_cap)),
        None => OrderedJsonValue::Null,
    };
    OrderedJsonValue::Object(vec![
        member(
            "state_tokens",
            OrderedJsonValue::SignedInteger(i64::from(coverage.state_token_count)),
        ),
        member(
            "state_tokens_read",
            OrderedJsonValue::SignedInteger(i64::from(coverage.state_tokens_read)),
        ),
        member(
            "state_was_cut",
            OrderedJsonValue::Boolean(coverage.state_was_cut()),
        ),
        member("option_token_cap", option_token_cap),
        member(
            "distinguishable_options",
            OrderedJsonValue::SignedInteger(i64::from(coverage.distinguishable_option_count)),
        ),
    ])
}

fn member(key: &str, value: OrderedJsonValue) -> OrderedJsonMember {
    OrderedJsonMember {
        key: key.to_owned(),
        value,
    }
}

/// A probability, score or confidence rounded to four decimal places.
fn reported_number(number: f64) -> OrderedJsonValue {
    OrderedJsonValue::Float(
        (number * REPORTED_DECIMAL_PLACES_SCALE).round() / REPORTED_DECIMAL_PLACES_SCALE,
    )
}

#[cfg(test)]
#[path = "system_one_response_tests.rs"]
mod tests;
