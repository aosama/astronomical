use std::path::Path;

use super::{
    binary_digests, listed_test_names, parse_artifacts, test_build_arguments,
    workspace_binary_build_arguments,
};
use crate::test_run::arguments::TestRunRequest;

fn arguments(command_line: &str) -> Vec<String> {
    command_line.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn should_build_the_selected_profile_and_features_without_executing_tests() {
    let request = TestRunRequest::parse(&arguments(
        "--package astronomical-acceptance --test engine_prompt_cache --features direct-mlx --profile full-debug",
    ))
    .expect("test request");
    assert_eq!(
        test_build_arguments(&request),
        arguments(
            "test --package astronomical-acceptance --no-run --message-format=json --test engine_prompt_cache --profile full-debug --features direct-mlx"
        )
    );
    assert!(
        workspace_binary_build_arguments(&request)
            .expect("whole-product binaries")
            .ends_with(&arguments("--profile full-debug"))
    );
}

#[test]
fn should_not_build_model_workers_for_library_only_tooling_contracts() {
    let request = TestRunRequest::parse(&arguments("--package astronomical-test-harness --lib"))
        .expect("library request");
    assert!(workspace_binary_build_arguments(&request).is_none());
}

#[test]
fn should_collect_exact_executables_without_assuming_the_cargo_artifact_layout() {
    let repository = tempfile::tempdir().expect("repository");
    let cargo_messages = [
        serde_json::json!({
            "reason": "compiler-artifact",
            "target": {"name": "library_tests", "kind": ["lib"]},
            "profile": {"test": true},
            "executable": "output/custom/build/package/fingerprint/out/test-executable"
        }),
        serde_json::json!({
            "reason": "compiler-artifact",
            "target": {"name": "astronomical-inference-worker", "kind": ["bin"]},
            "profile": {"test": false},
            "executable": repository.path().join("output/custom/worker")
        }),
        serde_json::json!({
            "reason": "compiler-artifact",
            "target": {"name": "dependency", "kind": ["lib"]},
            "profile": {"test": false},
            "executable": null
        }),
        serde_json::json!({"reason": "build-finished", "success": true}),
    ]
    .map(|message| message.to_string())
    .join("\n");
    let artifacts = parse_artifacts(&cargo_messages, repository.path()).expect("artifacts");
    assert_eq!(
        artifacts.tests,
        [repository
            .path()
            .join("output/custom/build/package/fingerprint/out/test-executable")]
    );
    assert_eq!(
        artifacts.binaries["astronomical-inference-worker"],
        repository.path().join("output/custom/worker")
    );
}

#[test]
fn should_reject_conflicting_binary_paths_and_malformed_cargo_messages() {
    let compiler_artifact = |executable_path: &str| {
        serde_json::json!({
            "reason": "compiler-artifact",
            "target": {"name": "worker", "kind": ["bin"]},
            "profile": {"test": false},
            "executable": executable_path
        })
        .to_string()
    };
    let conflicting_messages = format!(
        "{}\n{}",
        compiler_artifact("first/worker"),
        compiler_artifact("second/worker")
    );
    assert!(parse_artifacts(&conflicting_messages, Path::new("/")).is_err());
    assert!(parse_artifacts("{malformed", Path::new("/")).is_err());
}

#[test]
fn should_list_only_named_test_cases_and_reject_zero_match_execution_later() {
    assert_eq!(
        listed_test_names(
            "first::should_work: test\nsecond::should_work: test\n2 tests, 0 benchmarks"
        ),
        ["first::should_work", "second::should_work"]
    );
    assert!(listed_test_names("0 tests, 0 benchmarks").is_empty());
}

#[test]
fn should_digest_the_selected_executable_bytes_and_report_missing_binaries() {
    let executable_directory = tempfile::tempdir().expect("executable directory");
    let executable_path = executable_directory.path().join("worker");
    std::fs::write(&executable_path, b"").expect("empty executable fixture");
    let executable_paths = [("worker".to_owned(), executable_path.clone())].into();
    assert_eq!(
        binary_digests(&executable_paths).expect("executable digest")["worker"],
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    std::fs::rename(&executable_path, executable_directory.path().join("moved"))
        .expect("move fixture");
    assert!(binary_digests(&executable_paths).is_err());
}
