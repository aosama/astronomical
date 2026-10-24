use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use serial_test::serial;

use super::process::{OutputStyle, run_command};
use super::{TEST_BUDGET, create_run_directory, load_configuration, require_success};
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
