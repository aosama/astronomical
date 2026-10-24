use super::{TestRunRequest, TestTarget};

fn arguments(command_line: &str) -> Vec<String> {
    command_line.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn should_select_a_named_test_and_explicit_configuration() {
    let request = TestRunRequest::parse(&arguments(
        "--package astronomical-acceptance --test engine_prompt_cache --features direct-mlx --release --configuration cases.json --filter selected_case --exact --ignored",
    ))
    .expect("the explicit case should parse");
    assert_eq!(request.package, "astronomical-acceptance");
    assert_eq!(
        request.target,
        TestTarget::Integration("engine_prompt_cache".to_owned())
    );
    assert_eq!(request.profile.as_deref(), Some("release"));
    assert_eq!(request.features.as_deref(), Some("direct-mlx"));
    assert!(request.exact && request.ignored);
}

#[test]
fn should_accept_library_and_all_test_targets_with_custom_profiles() {
    let library_request = TestRunRequest::parse(&arguments(
        "--package astronomical-test-harness --lib --profile full-debug",
    ))
    .expect("custom profile");
    assert_eq!(library_request.target, TestTarget::Library);
    assert_eq!(library_request.profile.as_deref(), Some("full-debug"));
    let all_tests = TestRunRequest::parse(&arguments(
        "--package astronomical-acceptance --tests --release --ignored",
    ))
    .expect("all ignored cases");
    assert_eq!(all_tests.target, TestTarget::AllTests);
}

#[test]
fn should_reject_missing_values_conflicting_targets_and_unknown_options() {
    for command_line in [
        "",
        "--package",
        "--package selected",
        "--package selected --test",
        "--package selected --lib --test another",
        "--package selected --tests --lib",
        "--package selected --lib --release --profile full-debug",
        "--package selected --lib --configuration",
        "--package selected --lib --exact",
        "--package selected --lib --unknown",
        "--package selected --lib --ignored --ignored",
        "--package selected --package another --lib",
    ] {
        assert!(
            TestRunRequest::parse(&arguments(command_line)).is_err(),
            "unexpectedly accepted {command_line}"
        );
    }
}

#[test]
fn should_require_optimized_builds_for_ignored_performance_measurements() {
    assert!(
        TestRunRequest::parse(&arguments(
            "--package astronomical-acceptance --test performance_throughput --ignored",
        ))
        .is_err()
    );
    assert!(
        TestRunRequest::parse(&arguments(
            "--package astronomical-acceptance --test performance_throughput --release --ignored",
        ))
        .is_ok()
    );
}

#[test]
fn should_keep_exact_case_selection_when_listing_ignored_tests() {
    let request = TestRunRequest::parse(&arguments(
        "--package selected --lib --filter selected_case --exact --ignored",
    ))
    .expect("selected test");
    assert_eq!(
        request.listing_arguments(true),
        arguments("--list --format terse selected_case --exact --ignored")
    );
    assert!(
        !request
            .listing_arguments(false)
            .contains(&"--ignored".to_owned())
    );
}
