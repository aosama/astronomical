//! The GPU training loop on a learnable synthetic Kev task: the right
//! option is the one whose state starts highest. A poor starting head must
//! train to at least 80% on questions it never saw.

use astronomical_decision_training::anchored_adam::AdamSettings;
use astronomical_decision_training::head_training::{HEAD_TRAINING_SETTINGS, TrainingSettings};
use serial_test::serial;

use super::super::gpu_test_support::{RankedTask, ranked_task};
use super::*;
use crate::synthetic_weights_test_support::initialize_test_runtime;

struct SilentProgress;

impl TrainingProgress for SilentProgress {
    fn report(&mut self, _completed_batches: u64, _total_batches: u64) {}
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

const FAST_SETTINGS: TrainingSettings = TrainingSettings {
    adam: AdamSettings {
        learning_rate: 1e-2,
        ..HEAD_TRAINING_SETTINGS.adam
    },
    seeds: &[1],
    ..HEAD_TRAINING_SETTINGS
};

#[test]
#[serial]
fn should_learn_a_learnable_rule_on_the_gpu() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let RankedTask {
        training_set,
        served,
        correct_options,
    } = ranked_task(math);
    let settings = FAST_SETTINGS;
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

#[test]
#[serial]
fn should_train_identically_when_every_pass_is_split_across_many_steps() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let ranked = ranked_task(math);
    let settings = TrainingSettings {
        maximum_epochs: 3,
        ..FAST_SETTINGS
    };
    let whole_pass_runs = train_seeds_on_gpu(
        math,
        &ranked.training_set,
        &ranked.served,
        settings,
        &mut SilentProgress,
    )
    .unwrap();

    // A zero budget ends every step after one batch.
    let mut split_trainer = GpuTrainer::with_step_time_budget(settings, Duration::ZERO);
    let mut step_count = 0;
    let mut reported_counts = Vec::new();
    while let TrainerStep::Running {
        completed_batches,
        total_batches,
    } = split_trainer
        .advance(math, &ranked.training_set, &ranked.served)
        .unwrap()
    {
        step_count += 1;
        reported_counts.push(completed_batches);
        // 3 passes of 18 batches, one run.
        assert_eq!(total_batches, 54);
    }
    let split_runs = split_trainer.take_runs();

    // 280 questions in batches of 16 are 18 batches a pass.
    assert!(step_count > 3 * 18, "{step_count} steps");
    assert!(reported_counts.windows(2).all(|pair| pair[0] <= pair[1]));
    // One batch a step: the count moves on every step of a pass, so a
    // caller's time-left estimate never waits a whole pass for news.
    assert!(
        reported_counts
            .windows(2)
            .filter(|pair| pair[0] < pair[1])
            .count()
            >= 50
    );
    assert_eq!(reported_counts.last(), Some(&54));
    assert_eq!(split_runs[0].epochs_run, whole_pass_runs[0].epochs_run);
    assert_eq!(split_runs[0].best_epoch, whole_pass_runs[0].best_epoch);
    assert_eq!(
        split_runs[0].best_choosing_loss,
        whole_pass_runs[0].best_choosing_loss
    );
    for (split_parameter, whole_parameter) in split_runs[0]
        .best_parameters
        .iter()
        .zip(&whole_pass_runs[0].best_parameters)
    {
        assert_eq!(
            split_parameter.to_vec_f32().unwrap(),
            whole_parameter.to_vec_f32().unwrap()
        );
    }
}
