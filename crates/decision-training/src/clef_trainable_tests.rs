//! Unit tests for Clef's trainable residual scorer and final mix.

use astronomical_decision_contract::{FloatVector, HeadInputs};

use super::*;
use crate::test_support::{assert_gradient_matches_finite_differences, sequence_values};

fn clef_inputs(
    field_vectors: Vec<Vec<f32>>,
    option_vectors: Vec<Vec<f32>>,
    lexical_priors: Vec<f32>,
    cosines: Vec<f32>,
) -> HeadInputs {
    let mut fields = Vec::with_capacity(field_vectors.len());
    for field_vector in field_vectors {
        fields.push(FloatVector(field_vector));
    }
    let mut options = Vec::with_capacity(option_vectors.len());
    for option_vector in option_vectors {
        options.push(FloatVector(option_vector));
    }
    HeadInputs::Clef {
        field_vectors: fields,
        option_vectors: options,
        lexical_priors,
        cosines,
    }
}

fn three_options(width: usize) -> HeadInputs {
    clef_inputs(
        vec![
            sequence_values(width, 31),
            sequence_values(width, 32),
            sequence_values(width, 33),
        ],
        vec![
            sequence_values(width, 34),
            sequence_values(width, 35),
            sequence_values(width, 36),
        ],
        vec![0.4, -0.2, 0.1],
        vec![0.3, 0.9, -0.5],
    )
}

#[test]
fn should_mix_prior_agreement_and_residual_as_serving_does() {
    let head = ClefTrainableHead::new(1, 1);
    // W0 reads only the field: a = 1; residual = gelu(1) = 0.841 345.
    // prior = agreement = e^0 = 1 and gate = sigmoid(0) = 0.5.
    let parameters = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let inputs = clef_inputs(vec![vec![1.0]], vec![vec![2.0]], vec![0.3], vec![0.5]);
    let option_logits = head.option_logits(&parameters, &inputs);
    // 0.3 + 0.5 · (0.5 + 0.841 345) = 0.970 672.
    assert!(
        (option_logits[0] - 0.970_672).abs() < 1e-5,
        "{option_logits:?}"
    );
}

#[test]
fn should_match_finite_differences_for_every_parameter() {
    let head = ClefTrainableHead::new(3, 4);
    let mut parameters = sequence_values(head.layout().parameter_count, 37);
    // Scales well below the ceiling, where every scalar has a gradient.
    head.layout()
        .tensor("prior_logit_scale")
        .values_mut(&mut parameters)[0] = 0.4;
    head.layout()
        .tensor("joint_logit_scale")
        .values_mut(&mut parameters)[0] = 1.2;
    assert_gradient_matches_finite_differences(
        &head,
        &parameters,
        &three_options(3),
        &[0.5, -1.0, 0.8],
    );
}

#[test]
fn should_hold_the_scales_at_the_ceiling_and_stop_their_gradient_there() {
    let head = ClefTrainableHead::new(3, 4);
    let mut parameters = sequence_values(head.layout().parameter_count, 38);
    head.layout()
        .tensor("prior_logit_scale")
        .values_mut(&mut parameters)[0] = 9.0;
    let at_nine = head.option_logits(&parameters, &three_options(3));
    head.layout()
        .tensor("prior_logit_scale")
        .values_mut(&mut parameters)[0] = 12.0;
    assert_eq!(head.option_logits(&parameters, &three_options(3)), at_nine);
    let mut gradient = vec![0.0; parameters.len()];
    head.add_gradient(
        &parameters,
        &three_options(3),
        &[1.0, 1.0, 1.0],
        &mut gradient,
    );
    assert_eq!(
        head.layout().tensor("prior_logit_scale").values(&gradient)[0],
        0.0
    );
}

#[test]
fn should_refuse_inputs_whose_lists_disagree_or_whose_vectors_are_another_width() {
    let head = ClefTrainableHead::new(3, 4);
    let short_priors = clef_inputs(
        vec![vec![0.0; 3]],
        vec![vec![0.0; 3]],
        Vec::new(),
        vec![0.0],
    );
    assert_eq!(
        head.check_inputs(&short_priors),
        Err(HeadInputsMismatch::OptionCountDisagrees)
    );
    let wide = clef_inputs(vec![vec![0.0; 4]], vec![vec![0.0; 4]], vec![0.0], vec![0.0]);
    assert_eq!(
        head.check_inputs(&wide),
        Err(HeadInputsMismatch::WrongWidth {
            expected: 3,
            found: 4
        })
    );
    assert_eq!(head.check_inputs(&three_options(3)), Ok(3));
}
