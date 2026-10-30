//! Unit contracts for the pure derivation helpers of the straight
//! full-attention layer.

use super::{attention_head_dimensions, attention_scale, rotary_position_parameters};

#[test]
fn should_derive_attention_head_dimensions_from_validated_pack_dimensions() {
    let head_dimensions = attention_head_dimensions(32, 8, 128);
    assert_eq!(head_dimensions.query_head_count, 32);
    assert_eq!(head_dimensions.key_value_head_count, 8);
    assert_eq!(head_dimensions.attention_head_dimension, 128);
    assert_eq!(head_dimensions.attention_projection_head_dimension, 256);
}

#[test]
fn should_not_swap_query_and_key_value_head_counts() {
    let head_dimensions = attention_head_dimensions(8, 32, 128);
    assert_eq!(head_dimensions.query_head_count, 8);
    assert_eq!(head_dimensions.key_value_head_count, 32);
    assert_eq!(head_dimensions.attention_head_dimension, 128);
    assert_eq!(head_dimensions.attention_projection_head_dimension, 256);
}

#[test]
fn should_derive_all_zero_head_dimensions_for_a_zero_head_dimension() {
    let head_dimensions = attention_head_dimensions(0, 0, 0);
    assert_eq!(head_dimensions.query_head_count, 0);
    assert_eq!(head_dimensions.key_value_head_count, 0);
    assert_eq!(head_dimensions.attention_head_dimension, 0);
    assert_eq!(head_dimensions.attention_projection_head_dimension, 0);
}

#[test]
fn should_derive_rotary_parameters_from_theta_and_partial_rotary_bits() {
    let rotary_parameters =
        rotary_position_parameters(1_000_000.0_f32.to_bits(), 0.5_f32.to_bits(), 128);
    assert_eq!(rotary_parameters.base_frequency, 1_000_000.0);
    assert_eq!(rotary_parameters.rotary_dimension, 64);
}

#[test]
fn should_yield_a_zero_rotary_dimension_for_a_zero_partial_rotary_factor() {
    let rotary_parameters =
        rotary_position_parameters(1_000_000.0_f32.to_bits(), 0.0_f32.to_bits(), 128);
    assert_eq!(rotary_parameters.base_frequency, 1_000_000.0);
    assert_eq!(rotary_parameters.rotary_dimension, 0);
}

#[test]
fn should_yield_a_full_head_rotary_dimension_for_a_full_partial_rotary_factor() {
    let rotary_parameters =
        rotary_position_parameters(1_000_000.0_f32.to_bits(), 1.0_f32.to_bits(), 96);
    assert_eq!(rotary_parameters.rotary_dimension, 96);
}

#[test]
fn should_yield_a_zero_base_frequency_for_zero_theta_bits() {
    let rotary_parameters = rotary_position_parameters(0, 0.5_f32.to_bits(), 128);
    assert_eq!(rotary_parameters.base_frequency, 0.0);
    assert_eq!(rotary_parameters.rotary_dimension, 64);
}

#[test]
fn should_scale_attention_by_the_inverse_square_root_of_the_head_dimension() {
    let scaled_attention = attention_scale(128);
    assert!((scaled_attention - 1.0 / 128.0_f32.sqrt()).abs() < f32::EPSILON);
}

#[test]
fn should_yield_a_unit_attention_scale_for_a_one_dimension_head() {
    assert_eq!(attention_scale(1), 1.0);
}

#[test]
fn should_yield_an_infinite_attention_scale_for_a_zero_dimension_head() {
    assert!(attention_scale(0).is_infinite());
}
