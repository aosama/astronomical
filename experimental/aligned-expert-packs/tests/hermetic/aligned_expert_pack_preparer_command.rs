use std::time::Duration;

use tokio::{process::Command, time::timeout};

const COMMAND_TEST_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
async fn should_explain_the_explicit_model_preparation_command_without_mutation() {
    let command_path =
        std::env::var("CARGO_BIN_EXE_astronomical-experimental-aligned-expert-pack-preparer")
            .expect("Cargo should provide the experimental model-preparer executable path");

    let command_output = timeout(
        COMMAND_TEST_TIMEOUT,
        Command::new(command_path).arg("--help").output(),
    )
    .await
    .expect("the model-preparer help command should finish within 120 seconds")
    .expect("the model-preparer help command should execute");

    assert!(command_output.status.success());
    let standard_output = String::from_utf8(command_output.stdout)
        .expect("the model-preparer help output should use UTF-8");
    assert!(standard_output.contains("--model-directory PATH"));
    assert!(standard_output.contains("--dry-run"));
    assert!(standard_output.contains("--yes"));
    assert!(standard_output.contains("--replace"));
    assert!(command_output.stderr.is_empty());
}

#[tokio::test]
async fn should_reject_missing_unknown_and_repeated_arguments_with_usage_status() {
    let invalid_argument_sets = [
        Vec::<&str>::new(),
        vec!["--unknown"],
        vec!["--model-directory", "/not-used", "--dry-run", "--dry-run"],
    ];

    for invalid_arguments in invalid_argument_sets {
        let command_output = run_model_preparer(invalid_arguments).await;
        assert_eq!(
            command_output.status.code(),
            Some(2),
            "invalid command arguments should return usage status"
        );
        assert!(command_output.stdout.is_empty());
        assert!(!command_output.stderr.is_empty());
    }
}

#[tokio::test]
async fn should_reject_mutation_flags_combined_with_dry_run() {
    for mutation_flag in ["--yes", "--replace"] {
        let command_output = run_model_preparer(vec![
            "--model-directory",
            "/not-used",
            "--dry-run",
            mutation_flag,
        ])
        .await;

        assert_eq!(command_output.status.code(), Some(2));
        let standard_error = String::from_utf8(command_output.stderr)
            .expect("the model-preparer usage error should use UTF-8");
        assert!(standard_error.contains("cannot be combined with --dry-run"));
    }
}

async fn run_model_preparer(arguments: Vec<&str>) -> std::process::Output {
    let command_path =
        std::env::var("CARGO_BIN_EXE_astronomical-experimental-aligned-expert-pack-preparer")
            .expect("Cargo should provide the experimental model-preparer executable path");
    timeout(
        COMMAND_TEST_TIMEOUT,
        Command::new(command_path).args(arguments).output(),
    )
    .await
    .expect("the model-preparer command should finish within 120 seconds")
    .expect("the model-preparer command should execute")
}
