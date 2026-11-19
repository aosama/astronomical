//! The GPU objective against the CPU losses it mirrors.

use astronomical_decision_training::training_losses::{
    LABEL_SMOOTHING, negative_log_likelihood, shipped_agreement_loss, smoothed_cross_entropy,
};
use serial_test::serial;

use super::super::gpu_test_support::assert_all_close;
use super::super::question_batch::OptionLayout;
use super::*;
use astronomical_model_primitives::synthetic_weights_test_support::initialize_test_runtime;

#[test]
#[serial]
fn should_give_the_label_and_agreement_gradients_the_cpu_losses_give() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = OptionLayout::new(vec![2, 3]);
    let question_batch = QuestionBatch::upload(math, &layout.batch_rows(&[0, 1])).unwrap();
    let scores = [0.4_f32, -1.1, 2.3, 0.2, -0.7];
    let smoothed_target = [
        (LABEL_SMOOTHING / 2.0) as f32,
        (1.0 - LABEL_SMOOTHING + LABEL_SMOOTHING / 2.0) as f32,
    ];
    let shipped = [0.2_f32, 0.5, 0.3];
    let row_targets = RowTargets::upload(
        math,
        &[
            HostRow {
                target: &smoothed_target,
                temperature: 2.0,
                weight: 0.5,
            },
            HostRow {
                target: &shipped,
                temperature: 1.5,
                weight: 0.25,
            },
        ],
        question_batch.widest_option_count,
    )
    .unwrap();
    let gpu_gradients = option_score_gradients(
        math,
        &math.upload(&scores, &[5]).unwrap(),
        &question_batch,
        &row_targets,
    )
    .unwrap();
    let (_, label_gradients) = smoothed_cross_entropy(&scores[0..2], 2.0, 1);
    let shipped_wide = [0.2_f64, 0.5, 0.3];
    let (_, agreement_gradients) = shipped_agreement_loss(&scores[2..5], 1.5, &shipped_wide);
    let mut expected = Vec::new();
    for label_gradient in label_gradients {
        expected.push(label_gradient * 0.5);
    }
    for agreement_gradient in agreement_gradients {
        expected.push(agreement_gradient * 0.25);
    }
    assert_all_close(
        &math.download(&gpu_gradients).unwrap(),
        &expected,
        "score gradients",
    );
}

#[test]
#[serial]
fn should_measure_the_mean_surprise_at_the_right_answer() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = OptionLayout::new(vec![2, 3]);
    let question_batch = QuestionBatch::upload(math, &layout.batch_rows(&[0, 1])).unwrap();
    let scores = [0.4_f32, -1.1, 2.3, 0.2, -0.7];
    let row_targets = RowTargets::upload(
        math,
        &[
            HostRow {
                target: &[1.0, 0.0],
                temperature: 1.0,
                weight: 1.0,
            },
            HostRow {
                target: &[0.0, 0.0, 1.0],
                temperature: 2.0,
                weight: 1.0,
            },
        ],
        question_batch.widest_option_count,
    )
    .unwrap();
    let gpu_mean = mean_cross_entropy(
        math,
        &math.upload(&scores, &[5]).unwrap(),
        &question_batch,
        &row_targets,
    )
    .unwrap();
    let cpu_mean = (negative_log_likelihood(&scores[0..2], 1.0, 0)
        + negative_log_likelihood(&scores[2..5], 2.0, 2))
        / 2.0;
    assert!(
        (gpu_mean - cpu_mean).abs() < 1e-5,
        "GPU {gpu_mean}, CPU {cpu_mean}"
    );
}
