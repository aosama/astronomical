//! Unit tests for Laya's wording of questions, options and states.

use super::*;
use crate::test_support::{ordered_json, single_question};

#[test]
fn should_number_question_kinds_as_the_checkpoints_were_trained() {
    assert_eq!(laya_question_type_index(DecisionQuestionKind::Choice), 0);
    assert_eq!(laya_question_type_index(DecisionQuestionKind::Score), 1);
    assert_eq!(laya_question_type_index(DecisionQuestionKind::Noul), 2);
}

#[test]
fn should_open_a_question_with_its_kind_and_instructions() {
    let question = single_question(r#"{"q": {"type": "noul", "instructions": "Is it late?"}}"#);
    assert_eq!(
        laya_instruction_text(&question),
        Ok("noul question: Is it late?".to_owned())
    );
}

#[test]
fn should_render_structured_instructions_as_json_text() {
    let question =
        single_question(r#"{"q": {"type": "noul", "instructions": {"task":"classify","n":2}}}"#);
    assert_eq!(
        laya_instruction_text(&question),
        Ok(r#"noul question: {"task": "classify", "n": 2}"#.to_owned())
    );
}

#[test]
fn should_reject_a_question_without_instructions() {
    let question = single_question(r#"{"q": {"type": "noul"}}"#);
    assert_eq!(
        laya_instruction_text(&question),
        Err(LayaRenderingError::MissingInstructions {
            question: "q".to_owned(),
        })
    );
}

#[test]
fn should_render_a_choice_option_with_and_without_a_description() {
    let question = single_question(
        r#"{"q": {"type": "choice", "instructions": "Pick.",
            "criteria": {"billing": "refunds", "other": null, "zero": 0}}}"#,
    );
    assert_eq!(
        laya_option_texts(&question),
        ["billing: refunds", "other", "zero: 0"]
    );
}

#[test]
fn should_render_score_levels_with_their_index() {
    let question = single_question(
        r#"{"q": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}"#,
    );
    assert_eq!(
        laya_option_texts(&question),
        ["level 0: low", "level 1: high"]
    );
}

#[test]
fn should_render_a_noul_as_false_then_true_with_default_wording() {
    let question = single_question(r#"{"q": {"type": "noul", "instructions": "Holds?"}}"#);
    assert_eq!(
        laya_option_texts(&question),
        [
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        ]
    );
}

#[test]
fn should_render_a_noul_with_custom_wording_and_labels() {
    let question = single_question(
        r#"{"q": {"type": "noul", "instructions": "Holds?",
            "criteria": {"true": "it holds"},
            "labels": {"false": "absent", "true": "present"}}}"#,
    );
    assert_eq!(
        laya_option_texts(&question),
        [
            "absent: no, the statement does not hold",
            "present: it holds"
        ]
    );
}

#[test]
fn should_read_a_text_state_as_written_and_a_structured_state_as_json() {
    assert_eq!(
        laya_state_text(&ordered_json(r#""as written""#)),
        "as written"
    );
    assert_eq!(
        laya_state_text(&ordered_json(r#"{"b":1,"a":[2]}"#)),
        r#"{"b": 1, "a": [2]}"#
    );
}

#[test]
fn should_cut_only_a_conversation_state_from_the_left() {
    assert!(laya_truncates_state_from_the_left(&ordered_json("[]")));
    assert!(!laya_truncates_state_from_the_left(&ordered_json("{}")));
    assert!(!laya_truncates_state_from_the_left(&ordered_json(
        r#""text""#
    )));
}
