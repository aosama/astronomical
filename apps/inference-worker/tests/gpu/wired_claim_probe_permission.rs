//! The permission a wired-memory probe run needs. Kept apart from the probe,
//! which walks Metal's wired residency and builds on macOS only, so the
//! permission rule is checked on every platform.

use test_harness::test_run::TestRunConfiguration;

pub(super) fn require_probe_permission(
    configuration: &TestRunConfiguration,
) -> Result<(), &'static str> {
    if configuration.allow_large_wired_memory_probe {
        Ok(())
    } else {
        Err("set allow_large_wired_memory_probe to true in the test-run configuration")
    }
}

#[test]
fn should_reject_the_wired_claim_probe_without_explicit_permission() {
    assert!(require_probe_permission(&TestRunConfiguration::default()).is_err());
}

#[test]
fn should_accept_explicit_permission_for_the_wired_claim_probe() {
    let configuration = TestRunConfiguration {
        allow_large_wired_memory_probe: true,
        ..Default::default()
    };
    assert_eq!(require_probe_permission(&configuration), Ok(()));
}
