use super::{ACTIVATIONS_FEATURE_NAME, PROJECTED_FEATURE_NAME, model_description};
use crate::projection_shape::ProjectionShape;

fn contains_subsequence(encoded: &[u8], expected_bytes: &[u8]) -> bool {
    for window in encoded.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

#[test]
fn should_declare_both_features_by_the_names_the_bridge_binds() {
    let description = model_description(ProjectionShape::new(2, 3, 4).expect("valid shape"));

    assert!(contains_subsequence(
        &description,
        ACTIVATIONS_FEATURE_NAME.as_bytes()
    ));
    assert!(contains_subsequence(
        &description,
        PROJECTED_FEATURE_NAME.as_bytes()
    ));
}

#[test]
fn should_shape_the_input_by_input_channels_and_the_output_by_output_channels() {
    let description = model_description(ProjectionShape::new(5, 3, 7).expect("valid shape"));

    // ArrayFeatureType shape [1, C, 1, T] then dataType FLOAT16 (65552).
    let input_array_type = [
        0x08, 0x01, 0x08, 0x05, 0x08, 0x01, 0x08, 0x07, 0x10, 0x90, 0x80, 0x04,
    ];
    let output_array_type = [
        0x08, 0x01, 0x08, 0x03, 0x08, 0x01, 0x08, 0x07, 0x10, 0x90, 0x80, 0x04,
    ];
    assert!(contains_subsequence(&description, &input_array_type));
    assert!(contains_subsequence(&description, &output_array_type));
}

#[test]
fn should_describe_a_single_token_single_channel_projection() {
    let description = model_description(ProjectionShape::new(1, 1, 1).expect("valid shape"));

    assert!(contains_subsequence(
        &description,
        &[
            0x08, 0x01, 0x08, 0x01, 0x08, 0x01, 0x08, 0x01, 0x10, 0x90, 0x80, 0x04
        ]
    ));
}
