//! The GPU passes against the CPU oracle: the same scores and, for a loss
//! whose score gradients are fixed weights, the same gradient for every
//! parameter. The oracle's own gradients are pinned by finite differences
//! in `astronomical-decision-training`.

use astronomical_decision_contract::HeadInputs;
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_decision_training::clef_trainable::ClefTrainableHead;
use astronomical_decision_training::kev_trainable::KevTrainableHead;
use astronomical_decision_training::laya_trainable::LayaTrainableHead;
use astronomical_decision_training::trainable_head::TrainableHead;
use serial_test::serial;

use super::super::gpu_test_support::{
    assert_all_close, clef_questions, kev_questions, laya_questions, sequence_values,
};
use super::super::gpu_upload::{
    download_parameters, gpu_head_family, upload_head_inputs, upload_parameters,
};
use super::super::question_batch::QuestionBatch;
use super::*;
use astronomical_model_primitives::synthetic_weights_test_support::initialize_test_runtime;

/// Scores and gradients from the GPU and from the CPU oracle, for one family.
fn compare_with_oracle(
    family: TunableHeadFamily,
    oracle: &dyn TrainableHead,
    questions: &[HeadInputs],
) {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let layout = oracle.layout();
    let mut parameter_values = sequence_values(layout.parameter_count, 7);
    for parameter in &mut parameter_values {
        *parameter *= 0.5;
    }
    let mut borrowed_questions = Vec::new();
    for question in questions {
        borrowed_questions.push(question);
    }
    let (gpu_inputs, option_layout) =
        upload_head_inputs(math, family, &borrowed_questions).unwrap();
    let parameters = upload_parameters(math, layout, &parameter_values).unwrap();
    let gpu_family = gpu_head_family(family, layout);
    let all_rows: Vec<usize> = (0..questions.len()).collect();
    let question_batch = QuestionBatch::upload(math, &option_layout.batch_rows(&all_rows)).unwrap();
    let gpu_scores = gpu_family
        .option_scores(math, &parameters, &gpu_inputs, &question_batch)
        .unwrap();
    let mut cpu_scores = Vec::new();
    let mut cpu_gradient = vec![0.0_f32; layout.parameter_count];
    let mut score_weights = Vec::new();
    for (question_index, question) in questions.iter().enumerate() {
        cpu_scores.extend(oracle.option_logits(&parameter_values, question));
        let weights = sequence_values(
            oracle.check_inputs(question).unwrap(),
            500 + question_index as u64,
        );
        oracle.add_gradient(&parameter_values, question, &weights, &mut cpu_gradient);
        score_weights.extend(weights);
    }
    assert_all_close(&math.download(&gpu_scores).unwrap(), &cpu_scores, "scores");
    let score_gradients = math.upload(&score_weights, &[score_weights.len()]).unwrap();
    let gpu_gradients = gpu_family
        .parameter_gradients(
            math,
            &parameters,
            &gpu_inputs,
            &question_batch,
            &score_gradients,
        )
        .unwrap();
    assert_all_close(
        &download_parameters(math, &gpu_gradients).unwrap(),
        &cpu_gradient,
        "gradients",
    );
}

#[test]
#[serial]
fn should_score_and_differentiate_kev_as_the_cpu_oracle_does() {
    compare_with_oracle(
        TunableHeadFamily::Kev,
        &KevTrainableHead::new(3, 5),
        &kev_questions(5),
    );
}

#[test]
#[serial]
fn should_score_and_differentiate_laya_as_the_cpu_oracle_does() {
    compare_with_oracle(
        TunableHeadFamily::Laya,
        &LayaTrainableHead::new(6, 4),
        &laya_questions(6),
    );
}

#[test]
#[serial]
fn should_score_and_differentiate_clef_as_the_cpu_oracle_does() {
    compare_with_oracle(
        TunableHeadFamily::Clef,
        &ClefTrainableHead::new(3, 5),
        &clef_questions(3),
    );
}

#[test]
#[serial]
fn should_refuse_inputs_of_another_family() {
    let runtime = initialize_test_runtime();
    let math = GpuMath { runtime: &runtime };
    let kev = kev_questions(4);
    let mut borrowed = Vec::new();
    for question in &kev {
        borrowed.push(question);
    }
    assert!(upload_head_inputs(math, TunableHeadFamily::Laya, &borrowed).is_err());
}
