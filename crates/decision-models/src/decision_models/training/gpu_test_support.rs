//! Shared helpers for the GPU training tests: deterministic values, and
//! small inputs for each family with questions of 2, 3 and 4 options.

use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{FloatVector, HeadInputs};
use decision_training::kev_trainable::KevTrainableHead;
use decision_training::trainable_head::TrainableHead;
use mlx_bridge::MlxArray;

use super::gpu_math::GpuMath;
use super::gpu_training::{GpuTrainingSet, TrainingRow};
use super::gpu_upload::{gpu_head_family, upload_head_inputs, upload_parameters};
use super::trainable_scorer::FinalLayersScorer;

/// Values in [-1, 1] from a small linear congruential sequence.
pub fn sequence_values(count: usize, seed: u64) -> Vec<f32> {
    let mut sequence_state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        sequence_state = sequence_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit_value = (sequence_state >> 40) as f64 / (1_u64 << 24) as f64;
        values.push((unit_value * 2.0 - 1.0) as f32);
    }
    values
}

pub const OPTION_COUNTS: [usize; 3] = [2, 3, 4];

fn vectors(count: usize, width: usize, seed: u64) -> Vec<FloatVector> {
    let mut float_vectors = Vec::with_capacity(count);
    for vector_index in 0..count {
        float_vectors.push(FloatVector(sequence_values(
            width,
            seed * 100 + vector_index as u64,
        )));
    }
    float_vectors
}

pub fn kev_questions(hidden_size: usize) -> Vec<HeadInputs> {
    let mut questions = Vec::new();
    for (question_index, option_count) in OPTION_COUNTS.iter().enumerate() {
        questions.push(HeadInputs::Kev {
            decision_state: FloatVector(sequence_values(hidden_size, 900 + question_index as u64)),
            option_states: vectors(*option_count, hidden_size, 10 + question_index as u64),
        });
    }
    questions
}

pub fn laya_questions(marker_width: usize) -> Vec<HeadInputs> {
    let mut questions = Vec::new();
    for (question_index, option_count) in OPTION_COUNTS.iter().enumerate() {
        questions.push(HeadInputs::Laya {
            marker_states: vectors(*option_count, marker_width, 20 + question_index as u64),
        });
    }
    questions
}

pub fn clef_questions(field_width: usize) -> Vec<HeadInputs> {
    let mut questions = Vec::new();
    for (question_index, option_count) in OPTION_COUNTS.iter().enumerate() {
        let seed = 30 + question_index as u64;
        questions.push(HeadInputs::Clef {
            field_vectors: vectors(*option_count, field_width, seed),
            option_vectors: vectors(*option_count, field_width, seed + 50),
            lexical_priors: sequence_values(*option_count, seed + 70),
            cosines: sequence_values(*option_count, seed + 90),
        });
    }
    questions
}

/// GPU against CPU, within float32 rounding: every GPU matrix product goes
/// through `GpuMath::precise_matmul`, so the TF32 path MLX takes on GPUs
/// that have it (about 1e-3 relative) does not show here; what is left is
/// the different order sums are added in.
pub fn assert_all_close(gpu_values: &[f32], cpu_values: &[f32], what: &str) {
    assert_eq!(gpu_values.len(), cpu_values.len(), "{what}: lengths differ");
    for (value_index, (gpu_value, cpu_value)) in gpu_values.iter().zip(cpu_values).enumerate() {
        let allowed = 1e-4 + 1e-4 * cpu_value.abs();
        assert!(
            (gpu_value - cpu_value).abs() <= allowed,
            "{what}[{value_index}]: GPU {gpu_value}, CPU {cpu_value}"
        );
    }
}

/// The width of the ranked-option task's states.
const RANKED_HIDDEN_SIZE: usize = 6;

/// A Kev question whose right option is the one whose state starts highest.
fn ranked_question(question_seed: u64) -> (HeadInputs, usize) {
    let mut option_states = Vec::new();
    let mut best_option = 0;
    let mut best_lead = f32::NEG_INFINITY;
    for option_index in 0..3 {
        let option_state = sequence_values(RANKED_HIDDEN_SIZE, question_seed * 10 + option_index);
        if option_state[0] > best_lead {
            best_lead = option_state[0];
            best_option = option_index as usize;
        }
        option_states.push(decision_contract::FloatVector(option_state));
    }
    (
        HeadInputs::Kev {
            decision_state: decision_contract::FloatVector(sequence_values(
                RANKED_HIDDEN_SIZE,
                question_seed + 7_000,
            )),
            option_states,
        },
        best_option,
    )
}

/// The ranked-option task over 400 questions: 280 to train on, 60 to
/// choose with, the rest to judge; a poor head that starts small.
pub struct RankedTask {
    pub training_set: GpuTrainingSet,
    pub served: Vec<MlxArray>,
    pub correct_options: Vec<usize>,
}

pub fn ranked_task(math: GpuMath<'_>) -> RankedTask {
    let oracle = KevTrainableHead::new(4, RANKED_HIDDEN_SIZE);
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
        scorer: Box::new(FinalLayersScorer {
            family: gpu_head_family(TunableHeadFamily::Kev, oracle.layout()),
            head_inputs,
        }),
        option_layout,
        rows,
        training_rows: (0..280).collect(),
        choosing_rows: (280..340).collect(),
        check_rows: Vec::new(),
    };
    let served = upload_parameters(math, oracle.layout(), &served_values).unwrap();
    RankedTask {
        training_set,
        served,
        correct_options,
    }
}
