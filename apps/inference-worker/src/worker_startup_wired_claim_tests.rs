//! Boundary tests for the wired-residency claim decision at worker startup.

use std::ffi::OsStr;

use crate::worker_startup::wired_residency_claim_budget_bytes;
use crate::worker_startup::wired_residency_claim_is_enabled;

#[test]
fn should_claim_wired_residency_when_the_environment_is_unset() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert!(wired_residency_claim_is_enabled(None))
}

#[test]
fn should_not_claim_wired_residency_when_explicitly_disabled() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert!(!wired_residency_claim_is_enabled(Some(OsStr::new("0"))))
}

#[test]
fn should_keep_the_claim_for_the_legacy_opt_in_value() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert!(wired_residency_claim_is_enabled(Some(OsStr::new("1"))))
}

#[test]
fn should_keep_the_claim_for_an_unrecognized_value_rather_than_silently_disarming() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    assert!(wired_residency_claim_is_enabled(Some(OsStr::new("yes"))))
}

#[test]
fn should_cap_the_wire_claim_at_the_recommended_working_set() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // MLX rejects wired claims above Metal's recommendation, so a machine
    // ceiling above the recommendation claims only the recommendation —
    // the 48 GB incident shape (issue #1297).
    assert_eq!(
        wired_residency_claim_budget_bytes(42_949_672_960, 40_000_028_672),
        40_000_028_672
    );
}

#[test]
fn should_claim_the_full_ceiling_below_the_recommendation() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // A configured ceiling below the recommendation is claimed as-is: the
    // engine never plans above the budget, so the budget never needs to
    // exceed the ceiling.
    assert_eq!(
        wired_residency_claim_budget_bytes(38_654_705_664, 40_000_028_672),
        38_654_705_664
    );
}
