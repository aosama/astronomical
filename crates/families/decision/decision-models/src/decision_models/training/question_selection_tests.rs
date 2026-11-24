//! Which questions a training takes, and what it keeps about each.

use decision_contract::{DecisionAnswer, FloatVector, NoulAnswer, QuestionInputCoverage};
use decision_training::kev_trainable::KevTrainableHead;

use super::*;

const SOURCE: RowSource = RowSource {
    example_index: 4,
    question_index: 1,
};

fn kev_outcome(question_name: &str, distinguishable_option_count: u32) -> QuestionOutcome {
    QuestionOutcome {
        question_name: question_name.to_owned(),
        answer: DecisionAnswer::Noul(NoulAnswer {
            probability_true: 0.5,
            answer_confidence: 0.5,
        }),
        option_logits: vec![0.0, 0.0],
        applied_temperature: 1.0,
        input_coverage: QuestionInputCoverage {
            state_token_count: 9,
            state_tokens_read: 9,
            option_token_cap: None,
            distinguishable_option_count,
        },
        head_inputs: Some(HeadInputs::Kev {
            decision_state: FloatVector(vec![0.5; 3]),
            option_states: vec![FloatVector(vec![1.0; 3]), FloatVector(vec![-1.0; 3])],
        }),
        training_record: None,
    }
}

fn empty_selection<'read>() -> QuestionSelection<'read> {
    QuestionSelection {
        head_inputs: Vec::new(),
        row_sources: Vec::new(),
        served_logits: Vec::new(),
        rows: Vec::new(),
        training_rows: Vec::new(),
        choosing_rows: Vec::new(),
        check_rows: Vec::new(),
        question_rows: Vec::new(),
        indistinguishable_question_count: 0,
    }
}

#[test]
fn should_take_a_labelled_question_into_its_part() {
    let head = KevTrainableHead::new(2, 3);
    let outcome = kev_outcome("in_love", 2);
    let mut selection = empty_selection();
    let row = select_question(
        &mut selection,
        (QuestionRole::Choosing { correct_option: 1 }, SOURCE),
        &outcome,
        Some(&head),
    )
    .unwrap();
    assert_eq!(row, Some(0));
    assert_eq!(selection.choosing_rows, [0]);
    assert!(selection.training_rows.is_empty());
    assert_eq!(selection.row_sources, [SOURCE]);
    assert_eq!(selection.head_inputs.len(), 1);
}

#[test]
fn should_refuse_a_right_answer_beyond_the_questions_options() {
    let head = KevTrainableHead::new(2, 3);
    let outcome = kev_outcome("in_love", 2);
    let mut selection = empty_selection();
    // Option 2 of a two-option question would index past its scores.
    let refusal = select_question(
        &mut selection,
        (QuestionRole::Training { correct_option: 2 }, SOURCE),
        &outcome,
        Some(&head),
    );
    assert!(matches!(
        refusal,
        Err(DecisionFailure::InvalidRequest { .. })
    ));
    assert!(selection.rows.is_empty());
}

#[test]
fn should_leave_out_a_question_whose_options_read_alike_and_count_only_labelled_ones() {
    let head = KevTrainableHead::new(2, 3);
    let outcome = kev_outcome("in_love", 1);
    let mut selection = empty_selection();
    let labelled = QuestionRole::Judging { correct_option: 0 };
    assert_eq!(
        select_question(&mut selection, (labelled, SOURCE), &outcome, Some(&head)).unwrap(),
        None
    );
    assert_eq!(
        select_question(
            &mut selection,
            (QuestionRole::Check, SOURCE),
            &outcome,
            Some(&head)
        )
        .unwrap(),
        None
    );
    assert_eq!(selection.indistinguishable_question_count, 1);
}

#[test]
fn should_take_a_question_without_head_inputs_for_a_whole_model_training() {
    let mut outcome = kev_outcome("in_love", 2);
    outcome.head_inputs = None;
    let mut selection = empty_selection();
    let row = select_question(
        &mut selection,
        (QuestionRole::Training { correct_option: 0 }, SOURCE),
        &outcome,
        None,
    )
    .unwrap();
    assert_eq!(row, Some(0));
    assert_eq!(selection.training_rows, [0]);
    assert_eq!(selection.row_sources, [SOURCE]);
    assert!(selection.head_inputs.is_empty());
}

#[test]
fn should_refuse_a_final_layer_question_that_came_back_without_head_inputs() {
    let head = KevTrainableHead::new(2, 3);
    let mut outcome = kev_outcome("in_love", 2);
    outcome.head_inputs = None;
    let mut selection = empty_selection();
    let refusal = select_question(
        &mut selection,
        (QuestionRole::Training { correct_option: 0 }, SOURCE),
        &outcome,
        Some(&head),
    );
    assert!(matches!(
        refusal,
        Err(DecisionFailure::FatalExecution { .. })
    ));
}
