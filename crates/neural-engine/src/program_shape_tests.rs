use super::{ProgramShape, ProgramShapeError};

#[test]
fn should_count_the_mixer_output_projection_operations_of_one_prefill_chunk() {
    let mixer_output = ProgramShape::new(4096, 2048, 2048).expect("valid shape");

    assert_eq!(mixer_output.weight_element_count(), 8_388_608);
    assert_eq!(mixer_output.activation_element_count(), 8_388_608);
    assert_eq!(mixer_output.projected_element_count(), 4_194_304);
    assert_eq!(mixer_output.floating_point_operations(), 34_359_738_368.0);
}

#[test]
fn should_accept_a_single_token_single_channel_projection() {
    let smallest = ProgramShape::new(1, 1, 1).expect("one is a valid dimension");

    assert_eq!(smallest.floating_point_operations(), 2.0);
}

#[test]
fn should_reject_a_projection_with_no_tokens() {
    assert_eq!(
        ProgramShape::new(2048, 2048, 0),
        Err(ProgramShapeError::EmptyDimension {
            input_channels: 2048,
            output_channels: 2048,
            token_count: 0,
        })
    );
}

#[test]
fn should_reject_a_dimension_beyond_thirty_two_bits() {
    let too_wide = i32::MAX as usize + 1;

    assert_eq!(
        ProgramShape::new(too_wide, 1, 1),
        Err(ProgramShapeError::DimensionTooLarge {
            dimension: too_wide
        })
    );
}
