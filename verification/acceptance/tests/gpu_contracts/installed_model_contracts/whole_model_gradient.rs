//! A whole-model training's gradient is the loss's own slope, array by
//! array, on a real checkpoint and the backend it runs on.
//!
//! The parity check (`whole_model_training`) proves the trainable copy
//! scores as serving does, which says nothing about the backward pass. A
//! backend with a wrong backward kernel passes parity and then learns slowly
//! or not at all, exactly like a model that is hard to train. This journey
//! tells the two apart before any weight moves: on the positive control's
//! coded job, it compares every trainable array's gradient with a central
//! difference of the loss along a pseudo-random direction confined to that
//! array, over one batch of training lines, at the starting parameters.
//!
//! At the start every add-on's `up` is zero, so the `down` arrays have no
//! gradient yet and their loss change is second order: they are measured
//! and reported, and judged only if the loss moves after all. The `up`
//! arrays that move the loss measurably are judged, and each one's gradient
//! carries the backward pass through every layer above it. Written for the
//! CUDA backend (issue #2699); run it with `cargo
//! test-laya-whole-model-gradient --configuration PATH` naming Laya's
//! `model_directory`.

use std::time::{Duration, Instant};

use model_serving::decision_models::training::directional_gradient_check::{
    ArrayGradientCheck, GradientCheckProgress, GradientCheckStage,
};
use model_serving::{DecisionEngine, LayaDecisionEngine};
use test_harness::test_run::TestRunContext;

use super::whole_model_positive_control::coded_job;
use super::whole_model_training::{
    TEST_MLX_BUFFER_CACHE_LIMIT_BYTES, TEST_MLX_MEMORY_CEILING_BYTES,
};

/// One training batch's worth of lines, as the training loop takes them.
const CHECKED_ROW_COUNT: usize = 16;
/// The loss change per unit step above which one array is judged on its
/// own. Below it, float32 rounding through 28 layers decides the measured
/// change's last few percent: measured on Laya on MLX's CUDA backend (issue
/// #2699), the gap shrank with
/// the change (a median of 0.9% near 1e-5, 0.4% near 1e-4, 0.3% near 1e-3)
/// with no lean either way, while a wrong backward kernel is off at every
/// size.
const LARGE_CHANGE: f64 = 1e-3;
/// How far a large array's gradient may be from the loss's slope; the
/// central difference itself is within about 1% at the check's step size.
const ALLOWED_LARGE_ARRAY_GAP: f64 = 2e-2;
/// The most the judged arrays' median gap, and their mean signed gap (a
/// gradient scaled wrong leans one way), may reach.
const ALLOWED_TYPICAL_GAP: f64 = 2e-2;
/// Fewer judged arrays than this means the check judged almost nothing.
const MINIMUM_JUDGED_ARRAY_COUNT: usize = 50;

/// How often the check's progress is printed.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Prints each stage's progress, its share done and the time it has left,
/// every [`PROGRESS_INTERVAL`] and at each stage's end. Each stage keeps its
/// own clock, so the arrays' estimate does not count the reading before it.
struct PrintedProgress {
    stage: Option<GradientCheckStage>,
    stage_started_at: Instant,
    last_printed_at: Instant,
}

impl PrintedProgress {
    fn new() -> Self {
        Self {
            stage: None,
            stage_started_at: Instant::now(),
            last_printed_at: Instant::now(),
        }
    }
}

impl GradientCheckProgress for PrintedProgress {
    fn report(&mut self, stage: GradientCheckStage, completed: usize, total: usize) {
        if self.stage != Some(stage) {
            self.stage = Some(stage);
            self.stage_started_at = Instant::now();
        }
        let is_stage_end = completed == total;
        if !is_stage_end && self.last_printed_at.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_printed_at = Instant::now();
        let elapsed_seconds = self.stage_started_at.elapsed().as_secs_f64();
        let remaining_seconds =
            elapsed_seconds / completed.max(1) as f64 * (total - completed) as f64;
        eprintln!(
            "[whole-model-gradient] stage={stage:?} completed={completed}/{total} progress_percent={:.1} elapsed_seconds={elapsed_seconds:.0} eta_seconds={remaining_seconds:.0}",
            100.0 * completed as f64 / total.max(1) as f64
        );
    }
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_give_every_laya_array_the_gradient_of_its_loss() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let mut decision_engine = LayaDecisionEngine::new(
        "whole-model-gradient",
        &model_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate");
    decision_engine.load().expect("the checkpoint should load");

    let array_checks = decision_engine
        .check_starting_gradient(coded_job(), CHECKED_ROW_COUNT, &mut PrintedProgress::new())
        .expect("the gradient check should run");

    let mut judged_checks = Vec::new();
    for array_check in &array_checks {
        if array_check.is_measurable() {
            print_array_check(array_check);
            judged_checks.push(*array_check);
        }
    }
    let agreement = GradientAgreement::of(&judged_checks);
    eprintln!(
        "[whole-model-gradient] family=Laya arrays={} judged={} median_gap={:.4} mean_signed_gap={:+.4} large_arrays={}",
        array_checks.len(),
        judged_checks.len(),
        agreement.median_gap,
        agreement.mean_signed_gap,
        agreement.large_array_count
    );
    assert!(
        judged_checks.len() >= MINIMUM_JUDGED_ARRAY_COUNT,
        "only {} arrays moved the loss measurably",
        judged_checks.len()
    );
    assert!(
        agreement.wrong_large_arrays.is_empty(),
        "these arrays' gradients are not the loss's slope: {:?}",
        agreement.wrong_large_arrays
    );
    assert!(
        agreement.median_gap < ALLOWED_TYPICAL_GAP
            && agreement.mean_signed_gap.abs() < ALLOWED_TYPICAL_GAP,
        "the gradient strays from the loss's slope overall: {agreement:?}"
    );
}

fn print_array_check(array_check: &ArrayGradientCheck) {
    eprintln!(
        "[whole-model-gradient] array={} predicted={:.6e} measured={:.6e} relative_gap={:.4}",
        array_check.parameter_index,
        array_check.predicted_change,
        array_check.measured_change,
        array_check.relative_gap()
    );
}

/// How the judged arrays' gradients agree with the loss's slope.
#[derive(Debug)]
struct GradientAgreement {
    median_gap: f64,
    mean_signed_gap: f64,
    large_array_count: usize,
    wrong_large_arrays: Vec<ArrayGradientCheck>,
}

impl GradientAgreement {
    fn of(judged_checks: &[ArrayGradientCheck]) -> Self {
        let mut relative_gaps = Vec::with_capacity(judged_checks.len());
        let mut signed_gap_sum = 0.0_f64;
        let mut large_array_count = 0;
        let mut wrong_large_arrays = Vec::new();
        for array_check in judged_checks {
            relative_gaps.push(array_check.relative_gap());
            signed_gap_sum += array_check.signed_relative_gap();
            if array_check.change_size() >= LARGE_CHANGE || !array_check.is_within(f64::INFINITY) {
                large_array_count += 1;
                if !array_check.is_within(ALLOWED_LARGE_ARRAY_GAP) {
                    wrong_large_arrays.push(*array_check);
                }
            }
        }
        relative_gaps.sort_by(f64::total_cmp);
        Self {
            median_gap: relative_gaps
                .get(relative_gaps.len() / 2)
                .copied()
                .unwrap_or(f64::NAN),
            mean_signed_gap: signed_gap_sum / judged_checks.len().max(1) as f64,
            large_array_count,
            wrong_large_arrays,
        }
    }
}
