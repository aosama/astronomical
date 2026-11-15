//! Shared helpers for this crate's unit tests.
//!
//! States are lines from Romeo and Juliet, the repository's standard text
//! fixture, so no test depends on random text.

use astronomical_decision_contract::DecisionQuestionKind;

use crate::scored_question::ScoredQuestion;

/// Short Romeo and Juliet lines used as states.
pub(crate) const ROMEO_AND_JULIET_LINES: [&str; 6] = [
    "Two households, both alike in dignity, in fair Verona, where we lay our scene.",
    "But soft, what light through yonder window breaks? It is the east, and Juliet is the sun.",
    "O Romeo, Romeo, wherefore art thou Romeo?",
    "What's in a name? That which we call a rose by any other word would smell as sweet.",
    "Good night, good night! Parting is such sweet sorrow.",
    "A plague o' both your houses!",
];

pub(crate) fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected} within {tolerance}, got {actual}"
    );
}

/// Borrows every question, as the fitting and metric functions take them.
pub(crate) fn borrowed(scored_questions: &[ScoredQuestion]) -> Vec<&ScoredQuestion> {
    let mut borrowed_questions = Vec::with_capacity(scored_questions.len());
    for scored_question in scored_questions {
        borrowed_questions.push(scored_question);
    }
    borrowed_questions
}

/// One scored question with the given logits and right answer.
pub(crate) fn scored_question(
    kind: DecisionQuestionKind,
    option_logits: &[f32],
    correct_option_index: usize,
    state_group_key: u64,
) -> ScoredQuestion {
    ScoredQuestion {
        question_name: format!("question-{state_group_key}"),
        kind,
        option_logits: option_logits.to_vec(),
        shipped_temperature: 1.0,
        correct_option_index,
        state_group_key,
    }
}

/// Yes-or-no questions a model answers "true" with logit gap `logit_gap`,
/// where "true" is right in `true_per_ten` of every ten questions.
///
/// With a gap of 3.0 the model says "true" at about 0.95 while being right
/// 70% of the time: overconfident, so the best temperature is about 3.54
/// (the one that turns a gap of 3.0 into odds of 7 to 3).
pub(crate) fn noul_questions(
    question_count: usize,
    logit_gap: f32,
    true_per_ten: usize,
    first_state_key: u64,
) -> Vec<ScoredQuestion> {
    let mut noul_questions = Vec::with_capacity(question_count);
    for question_position in 0..question_count {
        let correct_option_index = usize::from(question_position % 10 < true_per_ten);
        noul_questions.push(scored_question(
            DecisionQuestionKind::Noul,
            &[0.0, logit_gap],
            correct_option_index,
            first_state_key + question_position as u64,
        ));
    }
    noul_questions
}

/// `ln(7 / 3)`: the logit gap at which a 70%-right model is exactly calibrated.
pub(crate) const CALIBRATED_SEVENTY_PERCENT_GAP: f32 = 0.847_297_9;
/// The temperature that makes a 3.0 logit gap say 70%: `3 / ln(7/3)`.
pub(crate) const BEST_TEMPERATURE_FOR_GAP_THREE: f64 = 3.540_660_8;

/// Deterministic values in [-1, 1] from a small linear congruential
/// sequence: reproducible weights and states for gradient checks, without a
/// random-number dependency.
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

/// `Σ weight_k · logit_k`: a loss whose gradient with respect to the logits
/// is exactly the weights, so a head's backward pass can be checked alone.
fn weighted_logit_sum(logits: &[f32], logit_weights: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (logit, logit_weight) in logits.iter().zip(logit_weights) {
        total += f64::from(*logit) * f64::from(*logit_weight);
    }
    total
}

/// Checks a head's hand-written gradient against central finite differences
/// for every parameter, under the loss `Σ weight_k · logit_k`.
pub(crate) fn assert_gradient_matches_finite_differences(
    trainable_head: &dyn crate::trainable_head::TrainableHead,
    parameters: &[f32],
    head_inputs: &astronomical_decision_contract::HeadInputs,
    logit_weights: &[f32],
) {
    let mut analytic_gradient = vec![0.0_f32; parameters.len()];
    trainable_head.add_gradient(
        parameters,
        head_inputs,
        logit_weights,
        &mut analytic_gradient,
    );
    // A step large enough that float32 rounding of the logits stays far
    // below the difference, small enough that curvature does not dominate.
    let step = 1e-2_f32;
    let mut probe_parameters = parameters.to_vec();
    for parameter_index in 0..parameters.len() {
        probe_parameters[parameter_index] = parameters[parameter_index] + step;
        let raised = weighted_logit_sum(
            &trainable_head.option_logits(&probe_parameters, head_inputs),
            logit_weights,
        );
        probe_parameters[parameter_index] = parameters[parameter_index] - step;
        let lowered = weighted_logit_sum(
            &trainable_head.option_logits(&probe_parameters, head_inputs),
            logit_weights,
        );
        probe_parameters[parameter_index] = parameters[parameter_index];
        let finite_difference = (raised - lowered) / (2.0 * f64::from(step));
        let analytic = f64::from(analytic_gradient[parameter_index]);
        assert!(
            (analytic - finite_difference).abs() <= 2e-3 + 2e-2 * finite_difference.abs(),
            "parameter {parameter_index}: analytic {analytic}, finite difference {finite_difference}"
        );
    }
}
