use astronomical_test_harness::test_run::TestRunConfiguration;

use super::require_large_wired_memory_probe_permission;

#[test]
fn should_accept_explicit_permission_without_initializing_mlx() {
    let run_configuration = TestRunConfiguration {
        allow_large_wired_memory_probe: true,
        ..TestRunConfiguration::default()
    };

    require_large_wired_memory_probe_permission(run_configuration.allow_large_wired_memory_probe);
}

#[test]
#[should_panic(expected = "allow_large_wired_memory_probe")]
fn should_refuse_the_probe_when_no_permission_was_selected() {
    require_large_wired_memory_probe_permission(
        TestRunConfiguration::default().allow_large_wired_memory_probe,
    );
}

#[test]
#[should_panic(expected = "allow_large_wired_memory_probe")]
fn should_refuse_an_explicit_permission_denial_without_initializing_mlx() {
    require_large_wired_memory_probe_permission(false);
}
