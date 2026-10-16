//! Boundary tests for the wired-residency claim decision at worker startup.

use std::ffi::OsStr;

use astronomical_test_harness::offline_denied_test;

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
