//! A prepared training's host-side decisions: whether the trainable copy
//! starts where the model is, which run wins, when memory growth stops a
//! training, and how trained scores map back to the questions asked.

use super::*;
use astronomical_decision_contract::{
    DecisionAnswer, FloatVector, HeadInputs, NoulAnswer, QuestionInputCoverage,
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
    }
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
    let Some(winning_position) = best_run_position(&runs) else {
        panic!("three finished runs must have a best");
    };
    assert_eq!(runs[winning_position].seed, 2);
    assert!(best_run_position(&[]).is_none());
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

#[test]
fn should_accept_scores_within_rounding_of_the_models_and_report_the_largest_gap() {
    let served_logits = vec![vec![1.0, -2.0], vec![8.0]];
    let trainable_scores = vec![vec![1.0005, -2.0], vec![8.002]];

    let largest_gap = largest_parity_gap(&trainable_scores, &served_logits).unwrap();

    assert!((largest_gap - 0.002).abs() < 1e-5, "{largest_gap}");
}

#[test]
fn should_refuse_a_trainable_copy_that_scores_away_from_the_model_and_name_the_worst_gap() {
    let served_logits = vec![vec![1.0, -2.0], vec![0.5]];
    // The first gap is over the line; the last is far further over it.
    let trainable_scores = vec![vec![1.004, -2.0], vec![0.75]];

    let refusal = largest_parity_gap(&trainable_scores, &served_logits).unwrap_err();

    assert!(refusal.to_string().contains("0.25000"), "{refusal}");
    assert!(
        refusal.to_string().contains("nothing was trained"),
        "{refusal}"
    );
}

#[test]
fn should_refuse_a_trainable_copy_whose_score_is_not_a_number() {
    assert!(largest_parity_gap(&[vec![f32::NAN]], &[vec![1.0]]).is_err());
}

#[test]
fn should_let_a_training_hold_a_little_more_memory_than_it_started_with() {
    let starting_memory_bytes = 600 << 20;

    assert!(check_memory_growth(starting_memory_bytes, starting_memory_bytes).is_ok());
    assert!(check_memory_growth(starting_memory_bytes + (2 << 30), starting_memory_bytes).is_ok());
    // Memory may also fall below where it started, as a reclaimed cache does.
    assert!(check_memory_growth(100 << 20, starting_memory_bytes).is_ok());
}

#[test]
fn should_stop_a_training_whose_memory_keeps_growing_and_say_by_how_much() {
    let starting_memory_bytes = 600 << 20;

    let refusal =
        check_memory_growth(starting_memory_bytes + (5 << 30), starting_memory_bytes).unwrap_err();

    assert!(refusal.to_string().contains("5.0 GB more"), "{refusal}");
}
