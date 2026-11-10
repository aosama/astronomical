//! Reading a decision model's reference fixture and an engine's answers in
//! the same shape, for the parity journeys of each decision-model family.
//!
//! Every family's fixture is JSON written by a Python reference generator,
//! with each question's probabilities in canonical option order; these
//! helpers fail loudly on a malformed fixture, since a parity journey that
//! silently compared nothing would prove nothing.

use astronomical_decision_contract::{DecisionAnswer, OrderedJsonValue};

pub(super) fn reference_field<'a>(
    reference: &'a OrderedJsonValue,
    key: &str,
) -> &'a OrderedJsonValue {
    reference
        .member(key)
        .unwrap_or_else(|| panic!("the reference fixture has a `{key}` field"))
}

pub(super) fn reference_array<'a>(
    reference: &'a OrderedJsonValue,
    key: &str,
) -> &'a [OrderedJsonValue] {
    reference_field(reference, key)
        .as_array()
        .unwrap_or_else(|| panic!("reference field `{key}` is an array"))
}

pub(super) fn reference_number(value: &OrderedJsonValue) -> f64 {
    match value {
        OrderedJsonValue::SignedInteger(integer) => *integer as f64,
        OrderedJsonValue::UnsignedInteger(integer) => *integer as f64,
        OrderedJsonValue::Float(float) => *float,
        other_value => panic!("expected a number, got {}", other_value.type_name()),
    }
}

pub(super) fn reference_numbers(reference: &OrderedJsonValue, key: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    for value in reference_array(reference, key) {
        numbers.push(reference_number(value));
    }
    numbers
}

/// The probabilities of an answer in canonical option order.
pub(super) fn answer_probabilities(answer: &DecisionAnswer) -> Vec<f64> {
    match answer {
        DecisionAnswer::Choice(choice_answer) => {
            let mut probabilities = Vec::new();
            for option_probability in &choice_answer.option_probabilities {
                probabilities.push(option_probability.probability);
            }
            probabilities
        }
        DecisionAnswer::Score(score_answer) => score_answer.level_probabilities.clone(),
        DecisionAnswer::Noul(noul_answer) => {
            vec![
                1.0 - noul_answer.probability_true,
                noul_answer.probability_true,
            ]
        }
    }
}

pub(super) fn most_probable_index(probabilities: &[f64]) -> usize {
    let mut most_probable_index = 0;
    for (option_index, probability) in probabilities.iter().enumerate() {
        if *probability > probabilities[most_probable_index] {
            most_probable_index = option_index;
        }
    }
    most_probable_index
}
