//! Unit tests for the System One response shape.

use super::*;
use crate::answer::{ChoiceAnswer, NoulAnswer, OptionProbability, ScoreAnswer};
use crate::outcome::QuestionInputCoverage;

fn outcome(question_name: &str, answer: DecisionAnswer) -> QuestionOutcome {
    QuestionOutcome {
        question_name: question_name.to_owned(),
        answer,
        option_logits: Vec::new(),
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 900,
            state_tokens_read: 300,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
    }
}

fn choice_outcome() -> QuestionOutcome {
    outcome(
        "team",
        DecisionAnswer::Choice(ChoiceAnswer {
            chosen_key: "billing".to_owned(),
            option_probabilities: vec![
                OptionProbability {
                    key: "support".to_owned(),
                    probability: 0.123_449,
                },
                OptionProbability {
                    key: "billing".to_owned(),
                    probability: 0.876_551,
                },
            ],
            distribution_confidence: 0.461_25,
            answer_confidence: 0.876_551,
        }),
    )
}

fn json_text(value: &OrderedJsonValue) -> String {
    serde_json::to_string(value).expect("a response serializes")
}

#[test]
fn should_write_a_choice_with_probabilities_in_option_order() {
    assert_eq!(
        json_text(&system_one_answer(&choice_outcome())),
        r#"{"type":"choice","choice":"billing","probabilities":{"support":0.1234,"billing":0.8766},"confidence":0.4613,"answer_confidence":0.8766,"input":{"state_tokens":900,"state_tokens_read":300,"state_was_cut":true,"option_token_cap":null,"distinguishable_options":2}}"#
    );
}

#[test]
fn should_write_a_score_with_its_legend_and_level_probabilities() {
    let score_outcome = outcome(
        "urgency",
        DecisionAnswer::Score(ScoreAnswer {
            expected_level: 1.25,
            level_legend: vec!["low".to_owned(), "mid".to_owned(), "high".to_owned()],
            level_probabilities: vec![0.125, 0.5, 0.375],
            distribution_confidence: 0.1,
            answer_confidence: 0.5,
        }),
    );
    let answer_text = json_text(&system_one_answer(&score_outcome));
    assert!(answer_text.starts_with(
        r#"{"type":"score","score":1.25,"legend":{"0":"low","1":"mid","2":"high"},"probabilities":{"0":0.125,"1":0.5,"2":0.375},"confidence":0.1,"answer_confidence":0.5,"#
    ));
}

#[test]
fn should_write_a_noul_as_the_probability_of_true() {
    let noul_outcome = outcome(
        "refund",
        DecisionAnswer::Noul(NoulAnswer {
            probability_true: 0.2,
            answer_confidence: 0.8,
        }),
    );
    let answer_text = json_text(&system_one_answer(&noul_outcome));
    assert!(
        answer_text
            .starts_with(r#"{"type":"noul","noul":0.2,"confidence":0.8,"answer_confidence":0.8,"#)
    );
}

#[test]
fn should_report_an_uncut_state_and_an_option_cap() {
    let mut uncut_outcome = choice_outcome();
    uncut_outcome.input_coverage = QuestionInputCoverage {
        state_token_count: 40,
        state_tokens_read: 40,
        option_token_cap: Some(4),
        distinguishable_option_count: 1,
    };
    let answer_text = json_text(&system_one_answer(&uncut_outcome));
    assert!(answer_text.ends_with(
        r#""input":{"state_tokens":40,"state_tokens_read":40,"state_was_cut":false,"option_token_cap":4,"distinguishable_options":1}}"#
    ));
}

#[test]
fn should_keep_answers_in_question_order_and_report_usage() {
    let later_alphabetically = choice_outcome();
    let mut earlier_alphabetically = choice_outcome();
    earlier_alphabetically.question_name = "area".to_owned();
    let response_text = json_text(&system_one_response(
        "laya",
        &[later_alphabetically, earlier_alphabetically],
        1_024,
    ));
    assert!(response_text.starts_with(r#"{"model":"laya","answers":{"team":{"#));
    assert!(response_text.find(r#""team""#) < response_text.find(r#""area""#));
    assert!(response_text.ends_with(r#""usage":{"input_tokens":1024,"output_tokens":0}}"#));
}

#[test]
fn should_write_an_empty_answers_object_for_no_outcomes() {
    assert_eq!(
        json_text(&system_one_response("laya", &[], 0)),
        r#"{"model":"laya","answers":{},"usage":{"input_tokens":0,"output_tokens":0}}"#
    );
}

#[test]
fn should_round_at_the_fourth_decimal_place_boundary() {
    assert_eq!(reported_number(0.123_44), OrderedJsonValue::Float(0.1234));
    assert_eq!(reported_number(0.123_46), OrderedJsonValue::Float(0.1235));
    assert_eq!(reported_number(1.0), OrderedJsonValue::Float(1.0));
    assert_eq!(reported_number(0.0), OrderedJsonValue::Float(0.0));
}
