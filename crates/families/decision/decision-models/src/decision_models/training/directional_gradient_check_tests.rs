use mlx_bridge::MlxArray;
use model_primitives::synthetic_weights_test_support::initialize_test_runtime;
use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::super::gpu_test_support::{RankedTask, ranked_task};
use super::super::question_batch::QuestionBatch;
use super::super::row_objective::RowTargets;
use super::super::trainable_scorer::TrainableScorer;
use super::*;

/// How a scorer's reported gradients differ from the true ones: each is what
/// a broken backward kernel looks like from outside.
#[derive(Clone, Copy)]
enum GradientAlteration {
    /// A lost or doubled scale.
    Doubled,
    /// A dropped term: zero where the loss moves.
    Zeroed,
    /// A kernel that writes NaN.
    NotANumber,
    /// A wrong sign.
    Negated,
    /// One array's gradient missing.
    LastOneMissing,
}

struct AlteredGradientScorer {
    true_scorer: Box<dyn TrainableScorer>,
    alteration: GradientAlteration,
}

impl TrainableScorer for AlteredGradientScorer {
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        self.true_scorer
            .option_scores(math, parameters, question_batch)
    }

    fn parameter_gradients(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
        row_targets: &RowTargets,
    ) -> GpuResult<Vec<MlxArray>> {
        let mut true_gradients =
            self.true_scorer
                .parameter_gradients(math, parameters, question_batch, row_targets)?;
        let factor = match self.alteration {
            GradientAlteration::Doubled => 2.0,
            GradientAlteration::Zeroed => 0.0,
            GradientAlteration::NotANumber => f32::NAN,
            GradientAlteration::Negated => -1.0,
            GradientAlteration::LastOneMissing => {
                true_gradients.pop();
                return Ok(true_gradients);
            }
        };
        let mut altered_gradients = Vec::with_capacity(true_gradients.len());
        for true_gradient in &true_gradients {
            altered_gradients.push(math.runtime.multiply_scalar(true_gradient, factor)?);
        }
        Ok(altered_gradients)
    }
}

/// Holds the scorer's place while the real one is moved into its wrapper.
struct PlaceholderScorer;

impl TrainableScorer for PlaceholderScorer {
    fn option_scores(
        &self,
        _math: GpuMath<'_>,
        _parameters: &[MlxArray],
        _question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        unreachable!("the placeholder scorer is replaced before any pass")
    }

    fn parameter_gradients(
        &self,
        _math: GpuMath<'_>,
        _parameters: &[MlxArray],
        _question_batch: &QuestionBatch,
        _row_targets: &RowTargets,
    ) -> GpuResult<Vec<MlxArray>> {
        unreachable!("the placeholder scorer is replaced before any pass")
    }
}

/// Records the reports, to show progress covers every array.
#[derive(Default)]
struct RecordedProgress {
    reports: Vec<(GradientCheckStage, usize, usize)>,
}

impl GradientCheckProgress for RecordedProgress {
    fn report(&mut self, stage: GradientCheckStage, completed: usize, total: usize) {
        self.reports.push((stage, completed, total));
    }
}

const CHECKED_BATCH_ROWS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

fn altered_task(math: GpuMath<'_>, alteration: GradientAlteration) -> RankedTask {
    let mut task = ranked_task(math);
    let true_scorer = std::mem::replace(&mut task.training_set.scorer, Box::new(PlaceholderScorer));
    task.training_set.scorer = Box::new(AlteredGradientScorer {
        true_scorer,
        alteration,
    });
    task
}

fn checks_of(math: GpuMath<'_>, task: &RankedTask) -> GpuResult<Vec<ArrayGradientCheck>> {
    directional_gradient_checks(
        math,
        &task.training_set,
        &task.served,
        &CHECKED_BATCH_ROWS,
        &mut RecordedProgress::default(),
    )
}

fn measurable_checks(array_checks: &[ArrayGradientCheck]) -> Vec<ArrayGradientCheck> {
    let mut judged_checks = Vec::new();
    for array_check in array_checks {
        if array_check.is_measurable() {
            judged_checks.push(*array_check);
        }
    }
    judged_checks
}

fn test_math(runtime: &MlxRuntime) -> GpuMath<'_> {
    GpuMath { runtime }
}

// --- positive ---

#[test]
#[file_serial]
fn should_find_a_correct_gradient_equal_to_the_slope_of_the_loss_on_every_array() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = ranked_task(math);
    let mut progress = RecordedProgress::default();

    let array_checks = directional_gradient_checks(
        math,
        &task.training_set,
        &task.served,
        &CHECKED_BATCH_ROWS,
        &mut progress,
    )
    .unwrap();

    assert_eq!(array_checks.len(), task.served.len());
    assert_eq!(
        progress.reports.last(),
        Some(&(
            GradientCheckStage::Checking,
            task.served.len(),
            task.served.len()
        ))
    );
    let judged_checks = measurable_checks(&array_checks);
    assert!(judged_checks.len() >= 2, "{array_checks:?}");
    for array_check in &judged_checks {
        assert!(array_check.is_within(2e-2), "{array_check:?}");
    }
}

// --- negative ---

#[test]
#[file_serial]
fn should_expose_a_gradient_twice_the_slope_of_the_loss() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = altered_task(math, GradientAlteration::Doubled);

    let judged_checks = measurable_checks(&checks_of(math, &task).unwrap());

    assert!(judged_checks.len() >= 2, "{judged_checks:?}");
    for array_check in &judged_checks {
        assert!(
            (array_check.signed_relative_gap() - 0.5).abs() < 2e-2,
            "a doubled gradient predicts twice the change: {array_check:?}"
        );
    }
}

#[test]
#[file_serial]
fn should_judge_and_fail_an_array_whose_gradient_was_dropped_while_the_loss_moves() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = altered_task(math, GradientAlteration::Zeroed);

    let judged_checks = measurable_checks(&checks_of(math, &task).unwrap());

    assert!(
        judged_checks.len() >= 2,
        "a dropped gradient must still be measured"
    );
    for array_check in &judged_checks {
        assert!(!array_check.is_within(0.5), "{array_check:?}");
    }
}

#[test]
#[file_serial]
fn should_judge_and_fail_every_array_whose_gradient_is_not_a_number() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = altered_task(math, GradientAlteration::NotANumber);

    let array_checks = checks_of(math, &task).unwrap();

    assert!(!array_checks.is_empty());
    for array_check in &array_checks {
        assert!(array_check.is_measurable(), "{array_check:?}");
        assert!(!array_check.is_within(1.0), "{array_check:?}");
    }
}

#[test]
#[file_serial]
fn should_expose_a_gradient_of_the_wrong_sign() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = altered_task(math, GradientAlteration::Negated);

    let judged_checks = measurable_checks(&checks_of(math, &task).unwrap());

    assert!(judged_checks.len() >= 2, "{judged_checks:?}");
    for array_check in &judged_checks {
        assert!(
            array_check.signed_relative_gap() < -1.9,
            "a negated gradient predicts the opposite change: {array_check:?}"
        );
    }
}

#[test]
#[file_serial]
fn should_refuse_a_scorer_that_returns_fewer_gradients_than_arrays() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = altered_task(math, GradientAlteration::LastOneMissing);

    let refusal = checks_of(math, &task).expect_err("every array needs its gradient");

    assert!(refusal.0.contains("gradients for"), "{refusal:?}");
}

#[test]
#[file_serial]
fn should_refuse_an_empty_batch() {
    let runtime = initialize_test_runtime();
    let math = test_math(&runtime);
    let task = ranked_task(math);

    let refusal = directional_gradient_checks(
        math,
        &task.training_set,
        &task.served,
        &[],
        &mut RecordedProgress::default(),
    )
    .expect_err("an empty batch has no loss to differentiate");

    assert!(
        refusal.0.contains("at least one training row"),
        "{refusal:?}"
    );
}

// --- edge / boundary ---

#[test]
fn should_not_judge_an_array_that_leaves_the_loss_unchanged() {
    let unchanged_array = ArrayGradientCheck {
        parameter_index: 3,
        predicted_change: 6e-11,
        measured_change: 0.0,
    };
    let barely_measurable_array = ArrayGradientCheck {
        predicted_change: MINIMUM_MEASURABLE_CHANGE,
        ..unchanged_array
    };
    assert!(!unchanged_array.is_measurable());
    assert!(barely_measurable_array.is_measurable());
}

#[test]
fn should_read_a_gradient_the_loss_ignores_as_wholly_wrong_not_infinitely_so() {
    let ignored_array = ArrayGradientCheck {
        parameter_index: 0,
        predicted_change: 1e-2,
        measured_change: 0.0,
    };
    let silent_array = ArrayGradientCheck {
        predicted_change: 0.0,
        ..ignored_array
    };
    assert_eq!(ignored_array.relative_gap(), 1.0);
    assert!(!ignored_array.is_within(0.5));
    assert_eq!(silent_array.relative_gap(), 0.0);
}

#[test]
fn should_size_a_direction_like_its_array() {
    let parameter_values = [3.0_f32, -4.0, 0.0, 5.0];
    let direction_values = pseudo_random_direction(&parameter_values, 7);
    let parameter_size = root_mean_square(&parameter_values);
    let direction_size = root_mean_square(&direction_values);
    // Four uniform draws scatter about their expected size; the scale is
    // what is checked, within a factor of two.
    assert!(
        direction_size > parameter_size / 2.0 && direction_size < parameter_size * 2.0,
        "direction size {direction_size} for an array of size {parameter_size}"
    );
}

#[test]
fn should_give_an_array_of_zeros_a_small_direction_of_its_own() {
    let direction_values = pseudo_random_direction(&[0.0; 4096], 3);
    let direction_size = root_mean_square(&direction_values);
    assert!(
        (direction_size - ZERO_ARRAY_DIRECTION_SIZE).abs() < ZERO_ARRAY_DIRECTION_SIZE * 0.05,
        "direction size {direction_size}"
    );
}

#[test]
fn should_draw_the_same_direction_for_the_same_array_every_time() {
    let parameter_values = [1.0_f32; 64];
    assert_eq!(
        pseudo_random_direction(&parameter_values, 11),
        pseudo_random_direction(&parameter_values, 11)
    );
    assert_ne!(
        pseudo_random_direction(&parameter_values, 11),
        pseudo_random_direction(&parameter_values, 12)
    );
}

fn root_mean_square(values: &[f32]) -> f32 {
    let mut squared_sum = 0.0_f32;
    for value in values {
        squared_sum += value * value;
    }
    (squared_sum / values.len() as f32).sqrt()
}
