use std::path::Path;

use crate::function_length_budget::function_length_clippy_command;

#[test]
fn should_run_the_unchanged_workspace_pass_when_nothing_is_left_out() {
    let clippy_command = function_length_clippy_command(Path::new("/workspace"), &[]);

    let clippy_arguments: Vec<_> = clippy_command.get_args().collect();
    assert_eq!(clippy_arguments, ["clippy", "--workspace", "--all-targets"]);
}

#[test]
fn should_exclude_each_left_out_package_from_the_workspace_pass() {
    let packages_left_out = ["apple-only".to_owned(), "engine".to_owned()];
    let clippy_command =
        function_length_clippy_command(Path::new("/workspace"), &packages_left_out);

    let clippy_arguments: Vec<_> = clippy_command.get_args().collect();
    assert_eq!(
        clippy_arguments,
        [
            "clippy",
            "--workspace",
            "--all-targets",
            "--exclude",
            "apple-only",
            "--exclude",
            "engine"
        ]
    );
}
