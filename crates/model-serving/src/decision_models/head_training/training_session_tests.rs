//! The session's host-side steps: which questions training takes, which run
//! wins, and how trained scores map back to the questions asked.

use astronomical_decision_contract::{
    DecisionAnswer, FloatVector, HeadInputs, NoulAnswer, QuestionInputCoverage,
};
use astronomical_decision_training::kev_trainable::KevTrainableHead;

use super::*;

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
    }
}

fn empty_selection<'read>() -> QuestionSelection<'read> {
    QuestionSelection {
        head_inputs: Vec::new(),
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
        QuestionRole::Choosing { correct_option: 1 },
        &outcome,
        &head,
    )
    .unwrap();
    assert_eq!(row, Some(0));
    assert_eq!(selection.choosing_rows, [0]);
    assert!(selection.training_rows.is_empty());
}

#[test]
fn should_refuse_a_right_answer_beyond_the_questions_options() {
    let head = KevTrainableHead::new(2, 3);
    let outcome = kev_outcome("in_love", 2);
    let mut selection = empty_selection();
    // Option 2 of a two-option question would index past its scores.
    let refusal = select_question(
        &mut selection,
        QuestionRole::Training { correct_option: 2 },
        &outcome,
        &head,
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
        select_question(&mut selection, labelled, &outcome, &head).unwrap(),
        None
    );
    assert_eq!(
        select_question(&mut selection, QuestionRole::Check, &outcome, &head).unwrap(),
        None
    );
    assert_eq!(selection.indistinguishable_question_count, 1);
}

fn seed_run(seed: u64, best_choosing_loss: f64) -> GpuSeedRun {
    GpuSeedRun {
        seed,
        best_parameters: Vec::new(),
        best_choosing_loss,
        best_epoch: 3,
        epochs_run: 9,
        ending: SeedEnding::StoppedImproving,
    }
}

#[test]
fn should_pick_the_lowest_choosing_loss_and_the_first_run_on_a_tie() {
    let runs = [seed_run(1, 0.5), seed_run(2, 0.25), seed_run(3, 0.25)];
    let Some(winning_run) = best_run(&runs) else {
        panic!("three finished runs must have a best");
    };
    assert_eq!(winning_run.seed, 2);
    assert!(best_run(&[]).is_none());
}

#[test]
fn should_map_trained_scores_back_and_leave_left_out_questions_empty() {
    let read_examples = vec![vec![kev_outcome("in_love", 2), kev_outcome("grieving", 1)]];
    let question_rows = vec![vec![Some(0), None]];
    let trained_scores = vec![vec![-0.5, 1.5]];
    let trained = trained_examples(&read_examples, &question_rows, &trained_scores);
    assert_eq!(trained[0][0].trained_logits, [-0.5, 1.5]);
    assert!(trained[0][1].trained_logits.is_empty());
    // What the final layers read stays in the worker.
    assert!(trained[0][0].served.head_inputs.is_none());
}
