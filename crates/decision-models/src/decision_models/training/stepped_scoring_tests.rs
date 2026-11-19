//! Scoring in small steps against scoring every question at once.

use std::time::Instant;

use serial_test::serial;

use super::super::gpu_math::GpuMath;
use super::super::gpu_test_support::ranked_task;
use super::super::gpu_training::question_scores;
use super::*;
use astronomical_model_primitives::synthetic_weights_test_support::initialize_test_runtime;

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

    let mut stepped_scoring = SteppedScoring::new(all_rows.clone(), question_count);
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
    let row_count = ranked.training_set.rows.len();
    let mut stepped_scoring = SteppedScoring::new((0..row_count).collect(), row_count);

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
    let row_count = ranked.training_set.rows.len();
    let mut stepped_scoring = SteppedScoring::new(Vec::new(), row_count);

    stepped_scoring
        .advance(math, &ranked.training_set, &ranked.served, Instant::now())
        .unwrap();

    assert!(stepped_scoring.is_finished());
    assert!(stepped_scoring.into_scores().iter().all(Vec::is_empty));
}

#[test]
#[serial]
fn should_score_only_the_chosen_rows_and_leave_the_others_empty() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let ranked = ranked_task(math);
    let row_count = ranked.training_set.rows.len();
    let all_rows: Vec<usize> = (0..row_count).collect();
    let whole_scores =
        question_scores(math, &ranked.training_set, &ranked.served, &all_rows).unwrap();
    let chosen_rows = vec![1, row_count - 1];
    let mut stepped_scoring = SteppedScoring::new(chosen_rows.clone(), row_count);

    stepped_scoring
        .advance(
            math,
            &ranked.training_set,
            &ranked.served,
            Instant::now() + std::time::Duration::from_secs(60),
        )
        .unwrap();

    let scores = stepped_scoring.into_scores();
    for row in 0..row_count {
        if chosen_rows.contains(&row) {
            assert_eq!(scores[row], whole_scores[row]);
        } else {
            assert!(scores[row].is_empty());
        }
    }
}
