use super::*;

#[test]
fn should_alternate_baseline_first_and_candidate_first_pairs() {
    assert_eq!(
        policy_order_for_trial(0),
        [
            WorkerDecodeExpertCachePolicy::Baseline,
            WorkerDecodeExpertCachePolicy::Candidate,
        ]
    );
    assert_eq!(
        policy_order_for_trial(1),
        [
            WorkerDecodeExpertCachePolicy::Candidate,
            WorkerDecodeExpertCachePolicy::Baseline,
        ]
    );
}

#[test]
fn should_report_a_zero_width_interval_for_identical_pair_improvements() {
    let confidence_interval = paired_relative_improvement_confidence_interval_95(
        &[100.0, 100.0, 100.0],
        &[90.0, 90.0, 90.0],
        true,
    )
    .expect("two or more pairs should produce an interval");

    assert_eq!(confidence_interval.mean_improvement_percent, 10.0);
    assert_eq!(confidence_interval.lower_bound_percent, 10.0);
    assert_eq!(confidence_interval.upper_bound_percent, 10.0);
}

#[test]
fn should_include_paired_variation_in_the_student_t_interval() {
    let confidence_interval = paired_relative_improvement_confidence_interval_95(
        &[100.0, 100.0, 100.0],
        &[90.0, 80.0, 70.0],
        true,
    )
    .expect("three complete pairs should produce an interval");

    assert_eq!(confidence_interval.mean_improvement_percent, 20.0);
    assert!(confidence_interval.lower_bound_percent < 0.0);
    assert!(confidence_interval.upper_bound_percent > 40.0);
}

#[test]
fn should_return_no_interval_for_insufficient_or_undefined_pairs() {
    assert!(paired_relative_improvement_confidence_interval_95(&[100.0], &[90.0], true).is_none());
    assert!(
        paired_relative_improvement_confidence_interval_95(&[0.0, 0.0], &[1.0, 0.0], true)
            .is_none()
    );
    assert!(
        paired_relative_improvement_confidence_interval_95(&[100.0, f64::NAN], &[90.0, 80.0], true)
            .is_none()
    );
}
