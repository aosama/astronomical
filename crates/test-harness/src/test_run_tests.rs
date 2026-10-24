use std::fs;
use std::io;
use std::path::PathBuf;

use super::{
    PagingExperimentCase, TEST_RUN_CONTEXT_FILE_NAME, TestRunConfiguration, TestRunContext,
};

#[test]
fn should_default_to_scripted_tests_without_diagnostics_or_memory_probe_permission() {
    let configuration = TestRunConfiguration::default();
    assert!(!configuration.real_models);
    assert!(!configuration.diagnostics);
    assert!(!configuration.allow_large_wired_memory_probe);
    assert!(!configuration.retain_isolated_home);
    assert!(configuration.validate().is_ok());
}

#[test]
fn should_convert_whole_gigabytes_using_the_binary_memory_convention() {
    let experiment = PagingExperimentCase::default();
    assert_eq!(
        experiment.maximum_mlx_memory_bytes().expect("valid memory"),
        32 * 1_073_741_824
    );
}

#[test]
fn should_reject_zero_or_overflowing_memory_limits() {
    for maximum_mlx_memory_gigabytes in [0, u64::MAX / 1_073_741_824 + 1] {
        let experiment = PagingExperimentCase {
            maximum_mlx_memory_gigabytes,
            ..PagingExperimentCase::default()
        };
        assert!(experiment.maximum_mlx_memory_bytes().is_err());
    }
}

#[test]
fn should_accept_the_largest_representable_whole_gigabyte_limit() {
    let experiment = PagingExperimentCase {
        maximum_mlx_memory_gigabytes: u64::MAX / 1_073_741_824,
        ..PagingExperimentCase::default()
    };
    assert!(experiment.maximum_mlx_memory_bytes().is_ok());
}

#[test]
fn should_reject_zero_experiment_token_counts_and_layer_intervals() {
    let invalid_experiments = [
        PagingExperimentCase {
            streaming_prompt_processing_chunk_tokens: 0,
            ..PagingExperimentCase::default()
        },
        PagingExperimentCase {
            prefill_graph_submission_layer_interval: 0,
            ..PagingExperimentCase::default()
        },
        PagingExperimentCase {
            generation_graph_submission_layer_interval: 0,
            ..PagingExperimentCase::default()
        },
    ];
    for paging_experiment in invalid_experiments {
        let configuration = TestRunConfiguration {
            paging_experiment,
            ..TestRunConfiguration::default()
        };
        assert!(configuration.validate().is_err());
    }
    assert!(
        TestRunConfiguration {
            prompt_processing_chunk_tokens: Some(0),
            ..TestRunConfiguration::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn should_reject_empty_paths_and_unknown_configuration_fields() {
    assert!(
        TestRunConfiguration {
            model_directory: Some(PathBuf::new()),
            ..TestRunConfiguration::default()
        }
        .validate()
        .is_err()
    );
    assert!(serde_json::from_str::<TestRunConfiguration>(r#"{"unknown_input": true}"#).is_err());
}

#[test]
fn should_resolve_all_relative_inputs_against_the_configuration_directory() {
    let configuration_directory = tempfile::tempdir().expect("configuration directory");
    let mut configuration = TestRunConfiguration {
        model_directory: Some(PathBuf::from("model")),
        evidence_directory: Some(PathBuf::from("evidence")),
        throughput_history_path: Some(PathBuf::from("throughput.jsonl")),
        coverage_inventory_output: Some(PathBuf::from("inventory")),
        ..TestRunConfiguration::default()
    };
    configuration.resolve_paths(configuration_directory.path());
    assert_eq!(
        configuration.model_directory,
        Some(configuration_directory.path().join("model"))
    );
    assert_eq!(
        configuration.evidence_directory,
        Some(configuration_directory.path().join("evidence"))
    );
    assert_eq!(
        configuration.throughput_history_path,
        Some(configuration_directory.path().join("throughput.jsonl"))
    );
    assert_eq!(
        configuration.coverage_inventory_output,
        Some(configuration_directory.path().join("inventory"))
    );
}

#[test]
fn should_round_trip_explicit_inputs_and_exact_workspace_executables() {
    let run_directory = tempfile::tempdir().expect("run directory");
    let mut run_context = TestRunContext {
        configuration: TestRunConfiguration {
            real_models: true,
            evidence_directory: Some(run_directory.path().join("evidence")),
            ..TestRunConfiguration::default()
        },
        ..TestRunContext::default()
    };
    run_context.workspace_binaries.insert(
        "astronomical-inference-worker".to_owned(),
        run_directory.path().join("worker"),
    );
    run_context
        .write_to_directory(run_directory.path())
        .expect("publish context");
    assert_eq!(
        TestRunContext::read_from_directory(run_directory.path()).expect("read context"),
        Some(run_context)
    );
}

#[test]
fn should_allow_plain_test_invocations_when_no_run_context_exists() {
    let run_directory = tempfile::tempdir().expect("run directory");
    assert_eq!(
        TestRunContext::read_from_directory(run_directory.path()).expect("missing context"),
        None
    );
}

#[test]
fn should_report_malformed_context_instead_of_using_default_inputs() {
    let run_directory = tempfile::tempdir().expect("run directory");
    fs::write(
        run_directory.path().join(TEST_RUN_CONTEXT_FILE_NAME),
        r#"{"configuration": {"unknown_input": true}}"#,
    )
    .expect("write malformed context");
    assert!(TestRunContext::read_from_directory(run_directory.path()).is_err());
}

#[test]
fn should_refuse_to_overwrite_an_existing_run_context() {
    let run_directory = tempfile::tempdir().expect("run directory");
    let run_context = TestRunContext::default();
    run_context
        .write_to_directory(run_directory.path())
        .expect("first context");
    let write_error = run_context
        .write_to_directory(run_directory.path())
        .expect_err("existing context must remain untouched");
    assert_eq!(write_error.kind(), io::ErrorKind::AlreadyExists);
}

#[test]
fn should_reject_unresolved_configuration_or_executable_paths_before_launch() {
    let run_directory = tempfile::tempdir().expect("run directory");
    let mut run_context = TestRunContext {
        configuration: TestRunConfiguration {
            model_directory: Some(PathBuf::from("relative-model")),
            ..TestRunConfiguration::default()
        },
        ..TestRunContext::default()
    };
    assert!(
        run_context
            .write_to_directory(run_directory.path())
            .is_err()
    );
    run_context.configuration.model_directory = None;
    run_context.workspace_binaries.insert(
        "astronomical-inference-worker".to_owned(),
        PathBuf::from("relative-worker"),
    );
    assert!(
        run_context
            .write_to_directory(run_directory.path())
            .is_err()
    );
}

#[test]
fn should_reject_malformed_or_unbound_workspace_executable_digests() {
    let run_directory = tempfile::tempdir().expect("run directory");
    let mut run_context = TestRunContext::default();
    run_context
        .workspace_binary_sha256
        .insert("worker".to_owned(), "ab".repeat(32));
    assert!(
        run_context
            .write_to_directory(run_directory.path())
            .is_err()
    );
    run_context
        .workspace_binaries
        .insert("worker".to_owned(), run_directory.path().join("worker"));
    run_context
        .workspace_binary_sha256
        .insert("worker".to_owned(), "not-a-digest".to_owned());
    assert!(
        run_context
            .write_to_directory(run_directory.path())
            .is_err()
    );
}

#[test]
#[ignore = "explicit runner contract; requires diagnostics=true in cargo xtask test-run configuration"]
fn should_receive_the_explicit_runner_configuration_and_generated_source_identity() {
    let run_context = TestRunContext::current();
    assert!(run_context.configuration.diagnostics);
    assert_eq!(
        run_context.configuration.prompt_processing_chunk_tokens,
        Some(2_048)
    );
    assert!(
        run_context
            .configuration
            .evidence_directory
            .as_ref()
            .is_some_and(|path| path.is_absolute())
    );
    assert!(
        run_context
            .source_commit
            .as_ref()
            .is_some_and(|commit| commit.len() >= 40)
    );
}
