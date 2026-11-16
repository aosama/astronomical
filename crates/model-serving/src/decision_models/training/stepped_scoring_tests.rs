//! Scoring in small steps against scoring every question at once.

use std::time::Instant;

use serial_test::serial;

use super::super::gpu_math::GpuMath;
use super::super::gpu_test_support::ranked_task;
use super::super::gpu_training::question_scores;
use super::*;
use crate::synthetic_weights_test_support::initialize_test_runtime;

#[test]
#[serial]
fn should_give_exactly_the_scores_of_one_pass_when_split_across_many_steps() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let ranked = ranked_task(math);
    let question_count = ranked.training_set.rows.len();
    let all_rows: Vec<usize> = (0..question_count).collect();
    let whole_scores =
        question_scores(math, &ranked.training_set, &ranked.served, &all_rows).unwrap();

    let mut stepped_scoring = SteppedScoring::new(question_count);
    let mut step_count = 0;
    while !stepped_scoring.is_finished() {
        // A deadline already past scores one call's questions per step.
        stepped_scoring
            .advance(math, &ranked.training_set, &ranked.served, Instant::now())
            .unwrap();
        step_count += 1;
    }

    assert_eq!(
        step_count,
        question_count.div_ceil(QUESTIONS_PER_SCORING_CALL)
    );
    assert_eq!(stepped_scoring.scored_question_count(), question_count);
    assert_eq!(stepped_scoring.into_scores(), whole_scores);
}

#[test]
#[serial]
fn should_score_everything_in_one_step_when_time_allows() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let ranked = ranked_task(math);
    let mut stepped_scoring = SteppedScoring::new(ranked.training_set.rows.len());

    stepped_scoring
        .advance(
            math,
            &ranked.training_set,
            &ranked.served,
            Instant::now() + std::time::Duration::from_secs(60),
        )
        .unwrap();

    assert!(stepped_scoring.is_finished());
}

#[test]
#[serial]
fn should_be_finished_at_once_with_no_questions() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let ranked = ranked_task(math);
    let mut stepped_scoring = SteppedScoring::new(0);

    stepped_scoring
        .advance(math, &ranked.training_set, &ranked.served, Instant::now())
        .unwrap();

    assert!(stepped_scoring.is_finished());
    assert!(stepped_scoring.into_scores().is_empty());
}
