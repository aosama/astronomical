//! Boundary tests for the wired-residency budget at worker startup.

use crate::worker_startup::wired_residency_claim_budget_bytes;

#[test]
fn should_cap_the_wired_residency_claim_at_the_recommended_working_set() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(wired_residency_claim_budget_bytes(96, 64), 64);
}

#[test]
fn should_claim_the_full_serving_ceiling_below_the_recommendation() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(wired_residency_claim_budget_bytes(32, 64), 32);
}

#[test]
fn should_not_claim_more_than_a_zero_recommendation() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(wired_residency_claim_budget_bytes(64, 0), 0);
}

#[test]
fn should_preserve_a_zero_serving_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(wired_residency_claim_budget_bytes(0, 64), 0);
}

#[test]
fn should_keep_equal_serving_and_recommendation_budgets() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(wired_residency_claim_budget_bytes(64, 64), 64);
}

#[test]
fn should_handle_the_platform_integer_boundary_without_overflow() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert_eq!(
        wired_residency_claim_budget_bytes(usize::MAX, usize::MAX),
        usize::MAX
    );
    assert_eq!(wired_residency_claim_budget_bytes(usize::MAX, 1), 1);
    assert_eq!(wired_residency_claim_budget_bytes(1, usize::MAX), 1);
}
