//! Boundary tests for the wired-residency budget at worker startup.

use super::wired_claim::wired_residency_claim_budget_bytes;

#[test]
fn should_cap_the_wired_residency_claim_at_the_recommended_working_set() {
    assert_eq!(wired_residency_claim_budget_bytes(96, 64), 64);
}

#[test]
fn should_claim_the_full_machine_ceiling_below_the_recommendation() {
    assert_eq!(wired_residency_claim_budget_bytes(32, 64), 32);
}

#[test]
fn should_not_claim_more_than_a_zero_recommendation() {
    assert_eq!(wired_residency_claim_budget_bytes(64, 0), 0);
}

#[test]
fn should_preserve_a_zero_machine_ceiling() {
    assert_eq!(wired_residency_claim_budget_bytes(0, 64), 0);
}

#[test]
fn should_keep_equal_machine_and_recommendation_budgets() {
    assert_eq!(wired_residency_claim_budget_bytes(64, 64), 64);
}

#[test]
fn should_handle_the_platform_integer_boundary_without_overflow() {
    assert_eq!(
        wired_residency_claim_budget_bytes(usize::MAX, usize::MAX),
        usize::MAX
    );
    assert_eq!(wired_residency_claim_budget_bytes(usize::MAX, 1), 1);
    assert_eq!(wired_residency_claim_budget_bytes(1, usize::MAX), 1);
}
