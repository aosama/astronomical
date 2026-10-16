//! Boundary tests for the wired-residency claim decision at worker startup.

use std::ffi::OsStr;

use astronomical_test_harness::offline_denied_test;

use crate::worker_startup::wired_residency_claim_budget_bytes;
use crate::worker_startup::wired_residency_claim_is_enabled;

offline_denied_test!(
    should_claim_wired_residency_when_the_environment_is_unset,
    { assert!(wired_residency_claim_is_enabled(None)) }
);

offline_denied_test!(should_not_claim_wired_residency_when_explicitly_disabled, {
    assert!(!wired_residency_claim_is_enabled(Some(OsStr::new("0"))))
});

offline_denied_test!(should_keep_the_claim_for_the_legacy_opt_in_value, {
    assert!(wired_residency_claim_is_enabled(Some(OsStr::new("1"))))
});

offline_denied_test!(
    should_keep_the_claim_for_an_unrecognized_value_rather_than_silently_disarming,
    // An unrecognized value must not fall through to "disabled": silently
    // dropping the claim returns the worker to the 2026-10-16 failure mode
    // where macOS paged the resident model and prefill crawled for tens of
    // seconds per chunk.
    { assert!(wired_residency_claim_is_enabled(Some(OsStr::new("yes")))) }
);

offline_denied_test!(should_cap_the_wire_claim_at_the_recommended_working_set, {
    // MLX rejects wired claims above Metal's recommendation, so a machine
    // ceiling above the recommendation claims only the recommendation —
    // the 48 GB incident shape (issue #1297).
    assert_eq!(
        wired_residency_claim_budget_bytes(42_949_672_960, 40_000_028_672),
        40_000_028_672
    );
});

offline_denied_test!(should_claim_the_full_ceiling_below_the_recommendation, {
    // A configured ceiling below the recommendation is claimed as-is: the
    // engine never plans above the budget, so the budget never needs to
    // exceed the ceiling.
    assert_eq!(
        wired_residency_claim_budget_bytes(38_654_705_664, 40_000_028_672),
        38_654_705_664
    );
});
