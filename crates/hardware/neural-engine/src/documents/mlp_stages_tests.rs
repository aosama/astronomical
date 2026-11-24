use super::validated_stage_shape;
use crate::documents::model_interface::ModelDocumentError;

#[test]
fn should_accept_weights_that_fill_the_stage_and_a_width_that_divides_its_input() {
    let stage_shape = validated_stage_shape(8, 4, 2, &[0; 32], 4).expect("the stage is consistent");

    assert_eq!(stage_shape.input_channels(), 8);
    assert_eq!(stage_shape.output_channels(), 4);
    assert_eq!(stage_shape.token_count(), 2);
}

#[test]
fn should_accept_a_part_width_equal_to_the_whole_input() {
    assert!(validated_stage_shape(8, 4, 2, &[0; 32], 8).is_ok());
}

#[test]
fn should_refuse_a_part_width_of_zero_or_one_that_does_not_divide_the_input() {
    for contraction_part_width in [0, 3, 16] {
        assert!(matches!(
            validated_stage_shape(8, 4, 2, &[0; 32], contraction_part_width),
            Err(ModelDocumentError::UnevenContractionPartWidth { .. })
        ));
    }
}

#[test]
fn should_refuse_weights_that_do_not_fill_the_stage() {
    assert!(matches!(
        validated_stage_shape(8, 4, 2, &[0; 31], 4),
        Err(ModelDocumentError::WeightCountMismatch {
            expected: 32,
            actual: 31
        })
    ));
}

#[test]
fn should_report_a_zero_dimension_as_having_no_place_for_weights() {
    assert!(matches!(
        validated_stage_shape(8, 0, 2, &[0; 4], 4),
        Err(ModelDocumentError::WeightCountMismatch {
            expected: 0,
            actual: 4
        })
    ));
}
