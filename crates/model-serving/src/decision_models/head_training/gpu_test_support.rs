//! Shared helpers for the GPU training tests: deterministic values, and
//! small inputs for each family with questions of 2, 3 and 4 options.

use astronomical_decision_contract::{FloatVector, HeadInputs};

/// Values in [-1, 1] from a small linear congruential sequence.
pub(crate) fn sequence_values(count: usize, seed: u64) -> Vec<f32> {
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

pub(crate) const OPTION_COUNTS: [usize; 3] = [2, 3, 4];

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

pub(crate) fn kev_questions(hidden_size: usize) -> Vec<HeadInputs> {
    let mut questions = Vec::new();
    for (question_index, option_count) in OPTION_COUNTS.iter().enumerate() {
        questions.push(HeadInputs::Kev {
            decision_state: FloatVector(sequence_values(hidden_size, 900 + question_index as u64)),
            option_states: vectors(*option_count, hidden_size, 10 + question_index as u64),
        });
    }
    questions
}

pub(crate) fn laya_questions(marker_width: usize) -> Vec<HeadInputs> {
    let mut questions = Vec::new();
    for (question_index, option_count) in OPTION_COUNTS.iter().enumerate() {
        questions.push(HeadInputs::Laya {
            marker_states: vectors(*option_count, marker_width, 20 + question_index as u64),
        });
    }
    questions
}

pub(crate) fn clef_questions(field_width: usize) -> Vec<HeadInputs> {
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
pub(crate) fn assert_all_close(gpu_values: &[f32], cpu_values: &[f32], what: &str) {
    assert_eq!(gpu_values.len(), cpu_values.len(), "{what}: lengths differ");
    for (value_index, (gpu_value, cpu_value)) in gpu_values.iter().zip(cpu_values).enumerate() {
        let allowed = 1e-4 + 1e-4 * cpu_value.abs();
        assert!(
            (gpu_value - cpu_value).abs() <= allowed,
            "{what}[{value_index}]: GPU {gpu_value}, CPU {cpu_value}"
        );
    }
}
