//! Unit tests for question outcomes and their wire form.

use super::*;
use crate::answer::{ChoiceAnswer, NoulAnswer, OptionProbability};

fn coverage(state_token_count: u32, state_tokens_read: u32) -> QuestionInputCoverage {
    QuestionInputCoverage {
        state_token_count,
        state_tokens_read,
        option_token_cap: None,
        distinguishable_option_count: 2,
    }
}

#[test]
fn should_report_a_state_as_cut_only_when_fewer_tokens_were_read() {
    assert!(coverage(900, 300).state_was_cut());
    assert!(!coverage(300, 300).state_was_cut());
    assert!(!coverage(0, 0).state_was_cut());
}

#[test]
fn should_survive_a_round_trip_through_json() {
    let outcome = QuestionOutcome {
        question_name: "mood".to_owned(),
        answer: DecisionAnswer::Choice(ChoiceAnswer {
            chosen_key: "grave".to_owned(),
            option_probabilities: vec![
                OptionProbability {
                    key: "grave".to_owned(),
                    probability: 0.75,
                },
                OptionProbability {
                    key: "merry".to_owned(),
                    probability: 0.25,
                },
            ],
            distribution_confidence: 0.19,
            answer_confidence: 0.75,
        }),
        option_logits: vec![1.5, 0.25],
        applied_temperature: 1.75,
        input_coverage: QuestionInputCoverage {
            state_token_count: 900,
            state_tokens_read: 300,
            option_token_cap: Some(4),
            distinguishable_option_count: 2,
        },
    };
    let json_text = serde_json::to_string(&outcome).expect("an outcome serializes");
    let restored: QuestionOutcome = serde_json::from_str(&json_text).expect("it parses back");
    assert_eq!(restored, outcome);
}

#[test]
fn should_tag_each_answer_kind_on_the_wire() {
    let noul_answer = DecisionAnswer::Noul(NoulAnswer {
        probability_true: 0.9,
        answer_confidence: 0.9,
    });
    let json_text = serde_json::to_string(&noul_answer).expect("an answer serializes");
    assert!(json_text.contains(r#""kind":"noul""#), "{json_text}");
}

#[test]
fn should_refuse_an_outcome_missing_a_field() {
    let incomplete = r#"{"question_name": "mood", "option_logits": []}"#;
    assert!(serde_json::from_str::<QuestionOutcome>(incomplete).is_err());
}
