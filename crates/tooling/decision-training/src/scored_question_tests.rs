//! Unit tests for checking a model outcome before it is used.

use decision_contract::{QuestionInputCoverage, answer_from_probabilities};

use super::*;
use crate::labelled_example::parse_labelled_example;
use crate::test_support::{ROMEO_AND_JULIET_LINES, assert_close};

fn speaker_question() -> LabelledQuestion {
    let line_text = format!(
        r#"{{"state": "{}", "questions": {{"speaker": {{"type": "choice", "instructions": "Who speaks?",
            "criteria": ["Romeo", "Juliet", "Nurse"]}}}}, "labels": {{"speaker": "Juliet"}}}}"#,
        ROMEO_AND_JULIET_LINES[2]
    )
    .replace('\n', " ");
    let mut labelled_example = parse_labelled_example(&line_text, 1).unwrap();
    labelled_example.labelled_questions.remove(0)
}

/// An outcome as an engine builds it: answer probabilities computed from
/// `option_logits` at `applied_temperature`, in canonical order.
fn consistent_outcome(
    labelled_question: &LabelledQuestion,
    option_logits: &[f32],
    applied_temperature: f64,
) -> QuestionOutcome {
    let option_probabilities = probabilities_at(option_logits, applied_temperature);
    QuestionOutcome {
        question_name: labelled_question.question.name.clone(),
        answer: answer_from_probabilities(&labelled_question.question, &option_probabilities)
            .unwrap(),
        option_logits: option_logits.to_vec(),
        applied_temperature,
        input_coverage: QuestionInputCoverage {
            state_token_count: 12,
            state_tokens_read: 12,
            option_token_cap: None,
            distinguishable_option_count: 3,
        },
        head_inputs: None,
        training_record: None,
    }
}

#[test]
fn should_keep_the_scores_of_an_outcome_whose_answer_matches_them() {
    let labelled_question = speaker_question();
    let question_outcome = consistent_outcome(&labelled_question, &[0.5, 2.0, -1.0], 1.6);
    let scored_question =
        scored_question_from_outcome(&labelled_question, &question_outcome, 42).unwrap();
    assert_eq!(scored_question.option_logits, vec![0.5, 2.0, -1.0]);
    assert_eq!(scored_question.correct_option_index, 1);
    assert_eq!(scored_question.kind, DecisionQuestionKind::Choice);
    assert_eq!(scored_question.state_group_key, 42);
    assert_close(scored_question.shipped_temperature, 1.6, 0.0);
}

#[test]
fn should_stop_when_the_scores_and_the_answer_disagree_on_option_order() {
    // The scores arrive reversed relative to the served answer: exactly the
    // mixup that would fit a temperature to the wrong options.
    let labelled_question = speaker_question();
    let mut question_outcome = consistent_outcome(&labelled_question, &[0.5, 2.0, -1.0], 1.0);
    question_outcome.option_logits = vec![-1.0, 2.0, 0.5];
    let scoring_outcome = scored_question_from_outcome(&labelled_question, &question_outcome, 1);
    assert!(matches!(
        scoring_outcome,
        Err(ScoredQuestionError::ScoresDisagreeWithAnswer {
            option_index: 0,
            ..
        })
    ));
}

#[test]
fn should_accept_scores_that_differ_from_the_answer_only_by_rounding() {
    let labelled_question = speaker_question();
    let mut question_outcome = consistent_outcome(&labelled_question, &[0.5, 2.0, -1.0], 1.0);
    // An engine that computed from f64 logits differs in the eighth decimal.
    question_outcome.option_logits = vec![0.500_000_1, 2.0, -1.0];
    assert!(scored_question_from_outcome(&labelled_question, &question_outcome, 1).is_ok());
}

#[test]
fn should_refuse_an_outcome_for_another_question() {
    let labelled_question = speaker_question();
    let mut question_outcome = consistent_outcome(&labelled_question, &[0.0, 1.0, 2.0], 1.0);
    question_outcome.question_name = "mood".to_owned();
    assert!(matches!(
        scored_question_from_outcome(&labelled_question, &question_outcome, 1),
        Err(ScoredQuestionError::AnsweredAnotherQuestion { .. })
    ));
}

#[test]
fn should_refuse_the_wrong_number_of_scores() {
    let labelled_question = speaker_question();
    let mut question_outcome = consistent_outcome(&labelled_question, &[0.0, 1.0, 2.0], 1.0);
    question_outcome.option_logits = vec![0.0, 1.0];
    assert!(matches!(
        scored_question_from_outcome(&labelled_question, &question_outcome, 1),
        Err(ScoredQuestionError::ScoreCountMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn should_refuse_a_score_that_is_not_a_finite_number() {
    let labelled_question = speaker_question();
    let mut question_outcome = consistent_outcome(&labelled_question, &[0.0, 1.0, 2.0], 1.0);
    question_outcome.option_logits = vec![0.0, f32::NAN, 2.0];
    assert!(matches!(
        scored_question_from_outcome(&labelled_question, &question_outcome, 1),
        Err(ScoredQuestionError::NonFiniteScore { .. })
    ));
}

#[test]
fn should_refuse_a_zero_or_non_finite_applied_temperature() {
    let labelled_question = speaker_question();
    for unusable_temperature in [0.0, -1.0, f64::INFINITY] {
        let mut question_outcome = consistent_outcome(&labelled_question, &[0.0, 1.0, 2.0], 1.0);
        question_outcome.applied_temperature = unusable_temperature;
        assert!(matches!(
            scored_question_from_outcome(&labelled_question, &question_outcome, 1),
            Err(ScoredQuestionError::UnusableTemperature { .. })
        ));
    }
}
