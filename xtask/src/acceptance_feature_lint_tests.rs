use std::path::Path;

use super::acceptance_feature_lint::{acceptance_feature_lint_command, run};

fn command_arguments(repository_root: &Path) -> Vec<String> {
    let lint_command = acceptance_feature_lint_command(repository_root);
    let mut command_arguments = Vec::new();
    for command_argument in lint_command.get_args() {
        command_arguments.push(command_argument.to_string_lossy().into_owned());
    }
    command_arguments
}

#[test]
fn should_lint_the_acceptance_tests_with_every_feature_on() {
    let lint_command = acceptance_feature_lint_command(Path::new("/repository"));

    assert_eq!(lint_command.get_program(), "cargo");
    assert_eq!(
        command_arguments(Path::new("/repository")),
        [
            "clippy",
            "--package",
            "astronomical-acceptance",
            "--tests",
            "--all-features"
        ]
    );
}

/// Naming features one by one would leave a feature added later unchecked,
/// which is the gap this command exists to close.
#[test]
fn should_not_name_individual_features() {
    let lint_arguments = command_arguments(Path::new("/repository"));

    assert!(
        !lint_arguments
            .iter()
            .any(|argument| argument == "--features")
    );
    assert!(
        !lint_arguments
            .iter()
            .any(|argument| argument == "--workspace")
    );
}

/// The hook may run from any directory of a checkout whose path holds
/// spaces; cargo must still resolve the workspace it was asked about.
#[test]
fn should_run_in_the_repository_root_even_when_its_path_holds_spaces() {
    let spaced_repository_root = Path::new("/Volumes/Work Disk/astronomical checkout");

    let lint_command = acceptance_feature_lint_command(spaced_repository_root);

    assert_eq!(lint_command.get_current_dir(), Some(spaced_repository_root));
}

#[test]
fn should_refuse_arguments_before_running_anything() {
    let refusal = run(
        Path::new("/nonexistent/repository"),
        &["--features".to_owned(), "direct-mlx".to_owned()],
    );

    assert_eq!(
        refusal,
        Err("lint-acceptance-features does not accept arguments".to_owned())
    );
}
