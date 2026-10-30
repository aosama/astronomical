use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serial_test::serial;

use super::process::{OutputStyle, run_command};
use super::{
    TEST_BUDGET, case_finished_log_line, create_run_directory, load_configuration,
    owned_run_evidence_directory, require_success,
};
use crate::process_signals::SignalPipe;

#[test]
fn should_resolve_explicit_configuration_inputs_against_the_configuration_file() {
    let configuration_directory = tempfile::tempdir().expect("configuration directory");
    fs::create_dir(configuration_directory.path().join("model")).expect("model directory");
    let configuration_path = configuration_directory.path().join("case.json");
    fs::write(
        &configuration_path,
        r#"{
            "model_directory": "model",
            "evidence_directory": "evidence",
            "throughput_history_path": "measurements.jsonl",
            "coverage_inventory_output": "inventory",
            "diagnostics": true
        }"#,
    )
    .expect("write configuration");
    let configuration = load_configuration(Some(&configuration_path)).expect("configuration");
    let configuration_root = configuration_directory
        .path()
        .canonicalize()
        .expect("canonical configuration directory");
    assert_eq!(
        configuration.model_directory,
        Some(configuration_root.join("model"))
    );
    assert_eq!(
        configuration.evidence_directory,
        Some(configuration_root.join("evidence"))
    );
    assert!(configuration.diagnostics);
}

#[test]
fn should_reject_missing_malformed_or_invalid_explicit_configuration() {
    let configuration_directory = tempfile::tempdir().expect("configuration directory");
    assert!(load_configuration(Some(&configuration_directory.path().join("missing"))).is_err());
    for invalid_json in [
        "{",
        r#"{"unknown_input": true}"#,
        r#"{"model_directory": "absent-model"}"#,
        r#"{"prompt_processing_chunk_tokens": 0}"#,
        r#"{"paging_experiment": {"maximum_mlx_memory_gigabytes": 0}}"#,
    ] {
        let configuration_path = configuration_directory.path().join("invalid.json");
        fs::write(&configuration_path, invalid_json).expect("write invalid configuration");
        assert!(
            load_configuration(Some(&configuration_path)).is_err(),
            "unexpectedly accepted {invalid_json}"
        );
    }
}

#[test]
fn should_create_distinct_owned_run_directories_and_preserve_existing_evidence() {
    let repository_directory = tempfile::tempdir().expect("repository");
    let first_run = create_run_directory(repository_directory.path()).expect("first run");
    let second_run = create_run_directory(repository_directory.path()).expect("second run");
    assert_ne!(first_run, second_run);
    fs::write(first_run.join("evidence"), "retained").expect("evidence");
    assert!(first_run.is_dir() && second_run.is_dir());
    assert_eq!(
        fs::read_to_string(first_run.join("evidence")).expect("retained evidence"),
        "retained"
    );
    assert_eq!(TEST_BUDGET, Duration::from_secs(120));
}

#[test]
#[serial]
fn should_capture_output_preserve_failure_status_and_stop_only_a_timed_out_command() {
    let run_directory = tempfile::tempdir().expect("run directory");
    let signals = SignalPipe::install().expect("signal pipe");
    let run_fixture = |shell_command: &str, log_name: &str, budget: Duration| {
        let mut fixture_command = Command::new("sh");
        fixture_command.args(["-c", shell_command]);
        run_command(
            &mut fixture_command,
            &run_directory.path().join(log_name),
            "fixture",
            budget,
            OutputStyle::Quiet,
            &signals,
        )
        .expect("supervised fixture")
    };
    let successful_output = run_fixture(
        "printf 'captured\\n'",
        "success.log",
        Duration::from_secs(2),
    );
    assert_eq!(successful_output.stdout, "captured\n");
    assert_eq!(successful_output.exit_code, 0);
    assert!(require_success(successful_output, "fixture").is_ok());
    let failed_output = run_fixture("exit 7", "failure.log", Duration::from_secs(2));
    assert_eq!(
        require_success(failed_output, "fixture")
            .err()
            .expect("failure should propagate")
            .exit_code,
        7
    );
    let timeout_output = run_fixture("sleep 30 & wait", "timeout.log", Duration::from_millis(50));
    assert_eq!(timeout_output.exit_code, 124);
    assert!(Path::new(&run_directory.path().join("timeout.log")).is_file());
}

#[test]
fn should_default_evidence_directory_to_the_owned_run_directory() {
    let run_directory = PathBuf::from("/test-runs/4242-172");
    assert_eq!(
        owned_run_evidence_directory(None, &run_directory).expect("default evidence directory"),
        PathBuf::from("/test-runs/4242-172/evidence")
    );
}

#[test]
fn should_nest_explicit_evidence_directory_under_the_run_name() {
    let owned_evidence_directory = owned_run_evidence_directory(
        Some(Path::new("/shared/evidence")),
        Path::new("/test-runs/4242-172"),
    )
    .expect("owned evidence directory");
    assert_eq!(
        owned_evidence_directory,
        PathBuf::from("/shared/evidence/4242-172")
    );
}

#[test]
fn should_reject_evidence_resolution_when_the_run_directory_has_no_name() {
    let root_error =
        owned_run_evidence_directory(Some(Path::new("/shared/evidence")), Path::new("/"))
            .expect_err("the filesystem root has no name");
    assert_eq!(root_error, "the owned run directory has no name");
    let dot_error =
        owned_run_evidence_directory(Some(Path::new("/shared/evidence")), Path::new("."))
            .expect_err("the dot path has no name");
    assert_eq!(dot_error, "the owned run directory has no name");
}

#[test]
fn should_report_mid_run_progress_and_eta_on_the_case_finished_line() {
    // 20 elapsed seconds across 5 finished cases extrapolates to 5 more
    // seconds per remaining case: 20 / 5 * 5 = 20 seconds of ETA.
    let log_line = case_finished_log_line(5, 10, 20.0);
    assert_eq!(
        log_line,
        "[test-run] status=case-finished completed=5 total=10 progress_percent=50.0 eta_seconds=20.0"
    );
}

#[test]
fn should_format_the_final_case_with_full_progress_and_zero_eta() {
    let log_line = case_finished_log_line(10, 10, 40.0);
    assert_eq!(
        log_line,
        "[test-run] status=case-finished completed=10 total=10 progress_percent=100.0 eta_seconds=0.0"
    );
}

#[test]
fn should_format_fractional_progress_rounded_to_one_decimal() {
    let log_line = case_finished_log_line(1, 3, 2.0);
    assert_eq!(
        log_line,
        "[test-run] status=case-finished completed=1 total=3 progress_percent=33.3 eta_seconds=4.0"
    );
}
