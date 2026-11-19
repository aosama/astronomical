//! Unit tests for Laya's trainable scorer.

use decision_contract::{FloatVector, HeadInputs};

use super::*;
use crate::test_support::{assert_gradient_matches_finite_differences, sequence_values};

fn laya_inputs(marker_states: Vec<Vec<f32>>) -> HeadInputs {
    let mut marker_vectors = Vec::with_capacity(marker_states.len());
    for marker_state in marker_states {
        marker_vectors.push(FloatVector(marker_state));
    }
    HeadInputs::Laya {
        marker_states: marker_vectors,
    }
}

#[test]
fn should_normalize_then_score_through_the_exact_gelu() {
    let head = LayaTrainableHead::new(2, 2);
    let parameters = vec![
        1.0, 1.0, // norm scale
        0.0, 0.0, // norm shift
        1.0, 0.0, 0.0, 1.0, // identity hidden weight
        0.0, 0.0, // hidden bias
        1.0, 1.0, // output weight
        0.5, // output bias
    ];
    // [1, 3] normalizes to about [-1, 1]; gelu(-1) + gelu(1) = erf(1/√2) = 0.682 689.
    let option_logits = head.option_logits(&parameters, &laya_inputs(vec![vec![1.0, 3.0]]));
    assert!(
        (option_logits[0] - 1.182_689).abs() < 1e-4,
        "{option_logits:?}"
    );
}

#[test]
fn should_match_finite_differences_for_every_parameter() {
    let head = LayaTrainableHead::new(4, 3);
    let parameters = sequence_values(head.layout().parameter_count, 21);
    let inputs = laya_inputs(vec![
        sequence_values(4, 22),
        sequence_values(4, 23),
        sequence_values(4, 24),
    ]);
    assert_gradient_matches_finite_differences(&head, &parameters, &inputs, &[1.1, -0.6, 0.3]);
}

#[test]
fn should_refuse_inputs_of_another_family_width_or_no_options() {
    let head = LayaTrainableHead::new(4, 3);
    assert!(matches!(
        head.check_inputs(&HeadInputs::Kev {
            decision_state: FloatVector(vec![0.0; 4]),
            option_states: vec![FloatVector(vec![0.0; 4])],
        }),
        Err(HeadInputsMismatch::WrongFamily { found: "Kev", .. })
    ));
    assert_eq!(
        head.check_inputs(&laya_inputs(vec![vec![0.0; 5]])),
        Err(HeadInputsMismatch::WrongWidth {
            expected: 4,
            found: 5
        })
    );
    assert_eq!(
        head.check_inputs(&laya_inputs(Vec::new())),
        Err(HeadInputsMismatch::NoOptions)
    );
}

#[test]
fn should_give_a_constant_marker_state_only_the_shift_and_biases() {
    // A constant vector normalizes to zero, so only the shift reaches the layers.
    let head = LayaTrainableHead::new(3, 2);
    let mut parameters = sequence_values(head.layout().parameter_count, 25);
    let constant_logit = head.option_logits(&parameters, &laya_inputs(vec![vec![2.0; 3]]))[0];
    head.layout()
        .tensor("scorer.layers.0.weight")
        .values_mut(&mut parameters)
        .copy_from_slice(&[9.0, -9.0, 9.0]);
    let rescaled_logit = head.option_logits(&parameters, &laya_inputs(vec![vec![2.0; 3]]))[0];
    assert!((constant_logit - rescaled_logit).abs() < 1e-5);
}
