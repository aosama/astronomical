use super::{ReferenceProjectionError, reference_projection};
use astronomical_neural_engine::program_shape::ProgramShape;

#[test]
fn should_project_channels_by_tokens_like_a_hand_computed_product() {
    // Two input channels, three output channels, two tokens.
    let shape = ProgramShape::new(2, 3, 2).expect("valid shape");
    let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let activations = [1.0, -1.0, 10.0, 0.5];

    let projected = reference_projection(shape, &weights, &activations).expect("projects");

    // Row o, column t: w[o][0]·a[0][t] + w[o][1]·a[1][t].
    assert_eq!(projected, vec![21.0, 0.0, 43.0, -1.0, 65.0, -2.0]);
}

#[test]
fn should_project_a_single_value() {
    let shape = ProgramShape::new(1, 1, 1).expect("valid shape");

    assert_eq!(reference_projection(shape, &[3.0], &[-2.0]), Ok(vec![-6.0]));
}

#[test]
fn should_reject_missing_weights() {
    let shape = ProgramShape::new(2, 2, 1).expect("valid shape");

    assert_eq!(
        reference_projection(shape, &[1.0; 3], &[1.0; 2]),
        Err(ReferenceProjectionError::WeightCountMismatch {
            expected: 4,
            actual: 3
        })
    );
}

#[test]
fn should_reject_missing_activations() {
    let shape = ProgramShape::new(2, 2, 3).expect("valid shape");

    assert_eq!(
        reference_projection(shape, &[1.0; 4], &[1.0; 5]),
        Err(ReferenceProjectionError::ActivationCountMismatch {
            expected: 6,
            actual: 5
        })
    );
}
