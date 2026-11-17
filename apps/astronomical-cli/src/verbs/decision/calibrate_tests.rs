//! Unit tests for the calibration journey's own steps. The fitting and its
//! judgement are tested in the training crate; scoring against a live model is
//! an acceptance run on the Development instance.

use astronomical_decision_contract::{
    QuestionInputCoverage, QuestionOutcome, answer_from_probabilities,
};
use astronomical_decision_training::holdout_split::state_group_key;
use astronomical_decision_training::labelled_example::parse_labelled_example;
use astronomical_decision_training::temperature_softmax::probabilities_at;

use super::*;

const ROMEO_LINE: &str = "But soft, what light through yonder window breaks?";

fn romantic_example() -> LabelledExample {
    let line_text = format!(
        r#"{{"state": "{ROMEO_LINE}", "questions": {{"romantic": {{"type": "noul", "instructions": "Is this romantic?"}}}}, "labels": {{"romantic": true}}}}"#
    );
    parse_labelled_example(&line_text, 12).unwrap()
}

fn outcome_for(question_name: &str, labelled_example: &LabelledExample) -> QuestionOutcome {
    let option_logits = [0.0_f32, 2.0];
    let question = &labelled_example.labelled_questions[0].question;
    QuestionOutcome {
        question_name: question_name.to_owned(),
        answer: answer_from_probabilities(question, &probabilities_at(&option_logits, 1.0))
            .unwrap(),
        option_logits: option_logits.to_vec(),
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 10,
            state_tokens_read: 10,
            option_token_cap: None,
            distinguishable_option_count: 2,
        },
        head_inputs: None,
        training_record: None,
    }
}

#[test]
fn should_score_an_example_whose_answers_match_its_questions() {
    let labelled_example = romantic_example();
    let scored_questions = scored_example(
        &labelled_example,
        &[outcome_for("romantic", &labelled_example)],
    )
    .unwrap();
    assert_eq!(scored_questions.len(), 1);
    assert_eq!(scored_questions[0].correct_option_index, 1);
    assert_eq!(
        scored_questions[0].state_group_key,
        state_group_key(&labelled_example.state)
    );
}

#[test]
fn should_name_the_line_of_an_example_the_model_answered_wrongly() {
    let labelled_example = romantic_example();
    let missing_answers = scored_example(&labelled_example, &[]);
    let wrong_question = scored_example(
        &labelled_example,
        &[outcome_for("tragic", &labelled_example)],
    );
    assert!(matches!(
        missing_answers,
        Err(DecisionToolError::AnswerCountMismatch {
            line_number: 12,
            asked: 1,
            answered: 0
        })
    ));
    assert!(matches!(
        wrong_question,
        Err(DecisionToolError::Scoring {
            line_number: 12,
            ..
        })
    ));
}

#[test]
fn should_send_each_example_as_written() {
    let labelled_example = romantic_example();
    let requests = example_requests(std::slice::from_ref(&labelled_example));
    assert_eq!(requests.len(), 1);
    assert_eq!(*requests[0].state, labelled_example.state);
    assert!(example_requests(&[]).is_empty());
}
