//! Unit tests for Kev's trainable pointer head.

use astronomical_decision_contract::{FloatVector, HeadInputs};

use super::*;
use crate::test_support::{assert_gradient_matches_finite_differences, sequence_values};

fn kev_inputs(decision_state: Vec<f32>, option_states: Vec<Vec<f32>>) -> HeadInputs {
    let mut option_vectors = Vec::with_capacity(option_states.len());
    for option_state in option_states {
        option_vectors.push(FloatVector(option_state));
    }
    HeadInputs::Kev {
        decision_state: FloatVector(decision_state),
        option_states: option_vectors,
    }
}

#[test]
fn should_score_as_the_served_head_scores() {
    // The same head as the serving test: q = [h0, h1 + 1], k = [h2 + 1, h0].
    let head = KevTrainableHead::new(2, 3);
    let parameters = vec![
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // q.weight
        0.0, 1.0, // q.bias
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, // k.weight
        1.0, 0.0, // k.bias
    ];
    let inputs = kev_inputs(
        vec![2.0, 3.0, 0.0],
        vec![vec![1.0, 0.0, 2.0], vec![0.0, 5.0, -1.0]],
    );
    let option_logits = head.option_logits(&parameters, &inputs);
    let scale = 1.0 / 2.0_f32.sqrt();
    // q = [2, 4]; k1 = [3, 1] → 10/√2; k2 = [0, 0] → 0.
    assert!((option_logits[0] - 10.0 * scale).abs() < 1e-6);
    assert!(option_logits[1].abs() < 1e-6);
}

#[test]
fn should_match_finite_differences_for_every_parameter() {
    let head = KevTrainableHead::new(3, 4);
    let parameters = sequence_values(head.layout().parameter_count, 11);
    let inputs = kev_inputs(
        sequence_values(4, 12),
        vec![
            sequence_values(4, 13),
            sequence_values(4, 14),
            sequence_values(4, 15),
        ],
    );
    assert_gradient_matches_finite_differences(&head, &parameters, &inputs, &[0.7, -1.2, 0.4]);
}

#[test]
fn should_refuse_inputs_of_another_family_width_or_no_options() {
    let head = KevTrainableHead::new(2, 3);
    let laya_inputs = HeadInputs::Laya {
        marker_states: vec![FloatVector(vec![0.0; 3])],
    };
    assert!(matches!(
        head.check_inputs(&laya_inputs),
        Err(HeadInputsMismatch::WrongFamily { found: "Laya", .. })
    ));
    assert_eq!(
        head.check_inputs(&kev_inputs(vec![0.0; 3], vec![vec![0.0; 4]])),
        Err(HeadInputsMismatch::WrongWidth {
            expected: 3,
            found: 4
        })
    );
    assert_eq!(
        head.check_inputs(&kev_inputs(vec![0.0; 3], Vec::new())),
        Err(HeadInputsMismatch::NoOptions)
    );
    assert_eq!(
        head.check_inputs(&kev_inputs(vec![0.0; 3], vec![vec![0.0; 3]; 4])),
        Ok(4)
    );
}
