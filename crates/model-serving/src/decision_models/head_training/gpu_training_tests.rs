//! The GPU training loop on a learnable synthetic Kev task: the right
//! option is the one whose state starts highest. A poor starting head must
//! train to at least 80% on questions it never saw.

use astronomical_decision_contract::HeadInputs;
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_decision_training::anchored_adam::AdamSettings;
use astronomical_decision_training::head_training::{HEAD_TRAINING_SETTINGS, TrainingSettings};
use astronomical_decision_training::kev_trainable::KevTrainableHead;
use astronomical_decision_training::trainable_head::TrainableHead;
use serial_test::serial;

use super::super::gpu_test_support::sequence_values;
use super::super::gpu_upload::{gpu_head_family, upload_head_inputs, upload_parameters};
use super::*;
use crate::synthetic_weights_test_support::initialize_test_runtime;

const HIDDEN_SIZE: usize = 6;

struct SilentProgress;

impl TrainingProgress for SilentProgress {
    fn report(&mut self, _completed_epochs: u64, _total_epochs: u64) {}
}

fn ranked_question(question_seed: u64) -> (HeadInputs, usize) {
    let mut option_states = Vec::new();
    let mut best_option = 0;
    let mut best_lead = f32::NEG_INFINITY;
    for option_index in 0..3 {
        let option_state = sequence_values(HIDDEN_SIZE, question_seed * 10 + option_index);
        if option_state[0] > best_lead {
            best_lead = option_state[0];
            best_option = option_index as usize;
        }
        option_states.push(astronomical_decision_contract::FloatVector(option_state));
    }
    (
        HeadInputs::Kev {
            decision_state: astronomical_decision_contract::FloatVector(sequence_values(
                HIDDEN_SIZE,
                question_seed + 7_000,
            )),
            option_states,
        },
        best_option,
    )
}

fn accuracy(scores: &[Vec<f32>], correct: &[usize]) -> f64 {
    let mut correct_count = 0;
    for (question_scores, correct_option) in scores.iter().zip(correct) {
        let mut picked = 0;
        for (option_index, score) in question_scores.iter().enumerate() {
            if *score > question_scores[picked] {
                picked = option_index;
            }
        }
        correct_count += usize::from(picked == *correct_option);
    }
    correct_count as f64 / correct.len() as f64
}

#[test]
#[serial]
fn should_learn_a_learnable_rule_on_the_gpu() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let oracle = KevTrainableHead::new(4, HIDDEN_SIZE);
    let mut served_values = sequence_values(oracle.layout().parameter_count, 41);
    for parameter in &mut served_values {
        *parameter *= 0.1;
    }
    let mut questions = Vec::new();
    let mut correct_options = Vec::new();
    for question_seed in 0..400 {
        let (head_inputs, correct_option) = ranked_question(question_seed);
        questions.push(head_inputs);
        correct_options.push(correct_option);
    }
    let mut borrowed = Vec::new();
    for question in &questions {
        borrowed.push(question);
    }
    let (head_inputs, option_layout) =
        upload_head_inputs(math, TunableHeadFamily::Kev, &borrowed).unwrap();
    let mut rows = Vec::new();
    for correct_option in &correct_options {
        rows.push(TrainingRow::Labelled {
            correct_option: *correct_option,
            temperature: 1.0,
        });
    }
    let training_set = GpuTrainingSet {
        family: gpu_head_family(TunableHeadFamily::Kev, oracle.layout()),
        head_inputs,
        option_layout,
        rows,
        training_rows: (0..280).collect(),
        choosing_rows: (280..340).collect(),
        check_rows: Vec::new(),
    };
    let served = upload_parameters(math, oracle.layout(), &served_values).unwrap();
    let settings = TrainingSettings {
        adam: AdamSettings {
            learning_rate: 1e-2,
            ..HEAD_TRAINING_SETTINGS.adam
        },
        seeds: &[1],
        ..HEAD_TRAINING_SETTINGS
    };
    let seed_runs =
        train_seeds_on_gpu(math, &training_set, &served, settings, &mut SilentProgress).unwrap();
    let judging_rows: Vec<usize> = (340..400).collect();
    let before = question_scores(math, &training_set, &served, &judging_rows).unwrap();
    let after = question_scores(
        math,
        &training_set,
        &seed_runs[0].best_parameters,
        &judging_rows,
    )
    .unwrap();
    let judged_correct = &correct_options[340..400];
    assert!(accuracy(&before, judged_correct) < 0.6);
    assert!(accuracy(&after, judged_correct) >= 0.8);
    assert!(seed_runs[0].best_epoch > 0);
}
