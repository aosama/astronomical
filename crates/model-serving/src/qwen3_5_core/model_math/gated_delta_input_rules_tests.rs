use mlx_bridge::MlxDtype;

use super::{
    GatedDeltaDtypeViolation, GatedDeltaHeadGeometry, GatedDeltaHeadGeometryViolation,
    check_gated_delta_dtypes,
};

/// Qwen3.5 linear attention: 16 key heads and 32 value heads, both of
/// dimension 128, one batch row.
fn qwen3_5_head_geometry() -> GatedDeltaHeadGeometry {
    GatedDeltaHeadGeometry {
        batch_size: 1,
        key_head_count: 16,
        key_head_dimension: 128,
        value_head_count: 32,
        value_head_dimension: 128,
    }
}

#[test]
fn should_accept_the_qwen3_5_head_geometry() {
    assert_eq!(qwen3_5_head_geometry().check(), Ok(()));
}

#[test]
fn should_accept_the_smallest_geometry_of_one_head_and_one_element() {
    let smallest_geometry = GatedDeltaHeadGeometry {
        batch_size: 1,
        key_head_count: 1,
        key_head_dimension: 1,
        value_head_count: 1,
        value_head_dimension: 1,
    };
    assert_eq!(smallest_geometry.check(), Ok(()));
}

#[test]
fn should_refuse_a_zero_or_negative_dimension() {
    let mut empty_batch = qwen3_5_head_geometry();
    empty_batch.batch_size = 0;
    let mut negative_value_dimension = qwen3_5_head_geometry();
    negative_value_dimension.value_head_dimension = -128;
    for non_positive_geometry in [empty_batch, negative_value_dimension] {
        assert_eq!(
            non_positive_geometry.check(),
            Err(GatedDeltaHeadGeometryViolation::NonPositiveDimension)
        );
    }
}

#[test]
fn should_refuse_zero_key_heads_as_non_positive_without_dividing_by_zero() {
    let mut no_key_heads = qwen3_5_head_geometry();
    no_key_heads.key_head_count = 0;
    assert_eq!(
        no_key_heads.check(),
        Err(GatedDeltaHeadGeometryViolation::NonPositiveDimension)
    );
}

#[test]
fn should_refuse_value_heads_that_do_not_split_evenly_over_key_heads() {
    let mut uneven_heads = qwen3_5_head_geometry();
    uneven_heads.value_head_count = 24;
    assert_eq!(
        uneven_heads.check(),
        Err(GatedDeltaHeadGeometryViolation::ValueHeadsNotMultipleOfKeyHeads)
    );
}

#[test]
fn should_expect_one_value_by_key_state_matrix_per_value_head() {
    let mut narrow_values = qwen3_5_head_geometry();
    narrow_values.value_head_dimension = 16;
    assert_eq!(
        narrow_values.expected_recurrent_state_shape(),
        [1, 32, 16, 128]
    );
}

#[test]
fn should_accept_every_supported_activation_dtype_with_a_float32_state() {
    for activation_dtype in [MlxDtype::Float16, MlxDtype::BFloat16, MlxDtype::Float32] {
        assert_eq!(
            check_gated_delta_dtypes(MlxDtype::Float32, [activation_dtype; 5]),
            Ok(())
        );
    }
}

#[test]
fn should_refuse_a_state_that_is_not_float32() {
    for other_state_dtype in [
        MlxDtype::Float16,
        MlxDtype::BFloat16,
        MlxDtype::Float64,
        MlxDtype::Int32,
    ] {
        assert_eq!(
            check_gated_delta_dtypes(other_state_dtype, [MlxDtype::BFloat16; 5]),
            Err(GatedDeltaDtypeViolation::RecurrentStateNotFloat32)
        );
    }
}

#[test]
fn should_refuse_any_one_unsupported_activation_dtype() {
    for broken_input_index in 0..5 {
        let mut activation_dtypes = [MlxDtype::BFloat16; 5];
        activation_dtypes[broken_input_index] = MlxDtype::Int32;
        assert_eq!(
            check_gated_delta_dtypes(MlxDtype::Float32, activation_dtypes),
            Err(GatedDeltaDtypeViolation::UnsupportedActivationDtype),
            "input {broken_input_index}"
        );
    }
}

#[test]
fn should_report_the_state_before_the_activations_when_both_are_wrong() {
    assert_eq!(
        check_gated_delta_dtypes(MlxDtype::BFloat16, [MlxDtype::Int32; 5]),
        Err(GatedDeltaDtypeViolation::RecurrentStateNotFloat32)
    );
}

#[test]
fn should_refuse_negative_key_heads_as_non_positive() {
    let mut negative_key_heads = qwen3_5_head_geometry();
    negative_key_heads.key_head_count = -16;
    assert_eq!(
        negative_key_heads.check(),
        Err(GatedDeltaHeadGeometryViolation::NonPositiveDimension)
    );
}

#[test]
fn should_refuse_fewer_value_heads_than_key_heads() {
    let mut fewer_value_heads = qwen3_5_head_geometry();
    fewer_value_heads.value_head_count = 8;
    assert_eq!(
        fewer_value_heads.check(),
        Err(GatedDeltaHeadGeometryViolation::ValueHeadsNotMultipleOfKeyHeads)
    );
}

#[test]
fn should_report_a_non_positive_dimension_before_uneven_heads() {
    let mut empty_and_uneven = qwen3_5_head_geometry();
    empty_and_uneven.batch_size = 0;
    empty_and_uneven.value_head_count = 24;
    assert_eq!(
        empty_and_uneven.check(),
        Err(GatedDeltaHeadGeometryViolation::NonPositiveDimension)
    );
}
