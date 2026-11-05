use super::{NumericalComparisonError, compare_values};

#[test]
fn should_report_no_error_for_identical_values() {
    let comparison = compare_values(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]).expect("comparable");

    assert_eq!(comparison.maximum_absolute_error, 0.0);
    assert_eq!(comparison.maximum_relative_error, 0.0);
    assert_eq!(comparison.mean_absolute_error, 0.0);
    assert!((comparison.cosine_similarity - 1.0).abs() < 1e-12);
    assert_eq!(comparison.non_finite_candidate_count, 0);
    assert_eq!(comparison.compared_value_count, 3);
}

#[test]
fn should_measure_absolute_relative_and_mean_error() {
    let comparison = compare_values(&[2.0, 4.0], &[2.5, 4.0]).expect("comparable");

    assert_eq!(comparison.maximum_absolute_error, 0.5);
    assert_eq!(comparison.maximum_relative_error, 0.25);
    assert_eq!(comparison.mean_absolute_error, 0.25);
}

#[test]
fn should_scale_the_worst_error_by_the_reference_rms() {
    // Reference RMS is sqrt((9 + 16) / 2) = sqrt(12.5).
    let comparison = compare_values(&[3.0, -4.0], &[3.0, -3.5]).expect("comparable");

    let expected = 0.5 / 12.5_f64.sqrt();
    assert!((comparison.maximum_error_relative_to_reference_rms - expected).abs() < 1e-12);
}

#[test]
fn should_report_no_scaled_error_for_two_zero_vectors() {
    let comparison = compare_values(&[0.0], &[0.0]).expect("comparable");

    assert_eq!(comparison.maximum_error_relative_to_reference_rms, 0.0);
}

#[test]
fn should_report_unbounded_scaled_error_against_a_zero_reference() {
    let comparison = compare_values(&[0.0], &[0.5]).expect("comparable");

    assert_eq!(
        comparison.maximum_error_relative_to_reference_rms,
        f64::INFINITY
    );
}

#[test]
fn should_skip_relative_error_where_the_reference_is_near_zero() {
    let comparison = compare_values(&[0.0, 1.0], &[0.01, 1.0]).expect("comparable");

    assert_eq!(comparison.maximum_relative_error, 0.0);
    assert!((comparison.maximum_absolute_error - 0.01).abs() < 1e-9);
}

#[test]
fn should_report_opposite_vectors_with_negative_cosine() {
    let comparison = compare_values(&[1.0, 2.0], &[-1.0, -2.0]).expect("comparable");

    assert!((comparison.cosine_similarity + 1.0).abs() < 1e-12);
}

#[test]
fn should_treat_two_zero_vectors_as_identical() {
    let comparison = compare_values(&[0.0, 0.0], &[0.0, 0.0]).expect("comparable");

    assert_eq!(comparison.cosine_similarity, 1.0);
}

#[test]
fn should_count_non_finite_candidates_without_poisoning_the_metrics() {
    let comparison =
        compare_values(&[1.0, 1.0, 1.0], &[f32::NAN, f32::INFINITY, 1.0]).expect("comparable");

    assert_eq!(comparison.non_finite_candidate_count, 2);
    assert_eq!(comparison.maximum_absolute_error, 0.0);
}

#[test]
fn should_reject_mismatched_lengths() {
    assert_eq!(
        compare_values(&[1.0], &[1.0, 2.0]),
        Err(NumericalComparisonError::LengthMismatch {
            reference_count: 1,
            candidate_count: 2
        })
    );
}

#[test]
fn should_reject_an_empty_comparison() {
    assert_eq!(
        compare_values(&[], &[]),
        Err(NumericalComparisonError::Empty)
    );
}
