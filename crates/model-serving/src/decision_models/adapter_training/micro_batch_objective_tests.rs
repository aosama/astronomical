//! The micro-batch objective against the formula final-layer training uses.

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};
use serial_test::serial;

use super::*;
use crate::decision_models::adapter_training::loss_gradient::{TrainableLoss, loss_gradients};
use crate::decision_models::head_training::gpu_math::GpuMath;
use crate::decision_models::head_training::row_objective::HostRow;
use crate::synthetic_weights_test_support::initialize_test_runtime;

/// Two questions in a batch three options wide: the first has two options
/// and a one-hot target, the second three and a soft target.
fn two_question_targets(math: GpuMath<'_>) -> RowTargets {
    RowTargets::upload(
        math,
        &[
            HostRow {
                target: &[0.0, 1.0],
                temperature: 0.5,
                weight: 0.25,
            },
            HostRow {
                target: &[0.2, 0.3, 0.5],
                temperature: 2.0,
                weight: 0.75,
            },
        ],
        3,
    )
    .unwrap()
}

/// `weight · −Σ target · log softmax(score / T)` for one row, in float64.
fn row_surprise(scores: &[f64], target: &[f64], temperature: f64, weight: f64) -> f64 {
    let tempered: Vec<f64> = scores.iter().map(|score| score / temperature).collect();
    let largest = tempered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_normalizer = largest
        + tempered
            .iter()
            .map(|tempered_score| (tempered_score - largest).exp())
            .sum::<f64>()
            .ln();
    let mut surprise = 0.0;
    for (tempered_score, target_probability) in tempered.iter().zip(target) {
        surprise -= target_probability * (tempered_score - log_normalizer);
    }
    weight * surprise
}

/// The objective as a function of the padded logits themselves.
struct LogitsAsTrainable {
    micro_batch_targets: MicroBatchTargets,
}

impl TrainableLoss for LogitsAsTrainable {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        self.micro_batch_targets
            .weighted_cross_entropy(context, &trainable_arrays[0])
    }
}

#[test]
#[serial]
fn should_weigh_each_questions_surprise_over_its_own_options_only() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let row_targets = two_question_targets(math);
    let micro_batch_targets =
        MicroBatchTargets::take(&runtime, &row_targets, &[0, 1], &[2, 3]).unwrap();
    // The first question's third slot is padding and holds a huge score
    // that must take no probability.
    let padded_logits = runtime
        .array_from_f32(&[1.0, 2.0, 99.0, 0.5, -1.0, 3.0], &[2, 3])
        .unwrap();

    let objective = micro_batch_targets
        .weighted_cross_entropy(&runtime, &padded_logits)
        .unwrap();

    let expected = row_surprise(&[1.0, 2.0], &[0.0, 1.0], 0.5, 0.25)
        + row_surprise(&[0.5, -1.0, 3.0], &[0.2, 0.3, 0.5], 2.0, 0.75);
    let measured = f64::from(objective.to_vec_f32().unwrap()[0]);
    assert!(
        (measured - expected).abs() < 1e-5,
        "{measured} vs {expected}"
    );
}

#[test]
#[serial]
fn should_widen_logits_narrower_than_the_batch_with_padding() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let row_targets = two_question_targets(math);
    // Only the first question, whose two options are all its micro-batch needs.
    let micro_batch_targets = MicroBatchTargets::take(&runtime, &row_targets, &[0], &[2]).unwrap();
    let narrow_logits = runtime.array_from_f32(&[1.0, 2.0], &[1, 2]).unwrap();

    let objective = micro_batch_targets
        .weighted_cross_entropy(&runtime, &narrow_logits)
        .unwrap();

    let expected = row_surprise(&[1.0, 2.0], &[0.0, 1.0], 0.5, 0.25);
    let measured = f64::from(objective.to_vec_f32().unwrap()[0]);
    assert!(
        (measured - expected).abs() < 1e-6,
        "{measured} vs {expected}"
    );
}

#[test]
#[serial]
fn should_give_the_same_score_gradient_final_layer_training_writes_by_hand() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let row_targets = two_question_targets(math);
    let micro_batch_targets =
        MicroBatchTargets::take(&runtime, &row_targets, &[0, 1], &[2, 3]).unwrap();
    let score_values = [1.0_f32, 2.0, 0.0, 0.5, -1.0, 3.0];
    let padded_logits = runtime.array_from_f32(&score_values, &[2, 3]).unwrap();

    let differentiated = loss_gradients(
        LogitsAsTrainable {
            micro_batch_targets,
        },
        &[padded_logits],
    )
    .unwrap();
    let score_gradients = differentiated.gradients[0].to_vec_f32().unwrap();

    // weight · (p − target) / T, row by row; padding gets none.
    let first_probabilities = [
        1.0 / (1.0 + (2.0_f64).exp()),
        (2.0_f64).exp() / (1.0 + (2.0_f64).exp()),
    ];
    let expected_first = [
        0.25 * (first_probabilities[0] - 0.0) / 0.5,
        0.25 * (first_probabilities[1] - 1.0) / 0.5,
        0.0,
    ];
    let second_exponentials = [(0.25_f64).exp(), (-0.5_f64).exp(), (1.5_f64).exp()];
    let second_total: f64 = second_exponentials.iter().sum();
    let second_targets = [0.2, 0.3, 0.5];
    let mut expected = expected_first.to_vec();
    for (exponential, target_probability) in second_exponentials.iter().zip(second_targets) {
        expected.push(0.75 * (exponential / second_total - target_probability) / 2.0);
    }
    for (option_position, (measured, wanted)) in score_gradients.iter().zip(&expected).enumerate() {
        assert!(
            (f64::from(*measured) - wanted).abs() < 1e-6,
            "option {option_position}: {measured} vs {wanted}"
        );
    }
}

#[test]
#[serial]
fn should_sum_micro_batch_gradients_and_start_from_the_first() {
    let runtime = initialize_test_runtime();
    let mut summed_gradients = Vec::new();

    add_and_evaluate(
        &runtime,
        &mut summed_gradients,
        vec![runtime.array_from_f32(&[1.0, 2.0], &[2]).unwrap()],
    )
    .unwrap();
    add_and_evaluate(
        &runtime,
        &mut summed_gradients,
        vec![runtime.array_from_f32(&[0.5, -4.0], &[2]).unwrap()],
    )
    .unwrap();

    assert_eq!(summed_gradients[0].to_vec_f32().unwrap(), vec![1.5, -2.0]);
}
