use std::path::Path;

use super::menu_contracts::{menu_contract_command, run};

#[test]
fn should_run_swift_test_on_the_menu_package_with_warnings_as_errors() {
    let swift_command = menu_contract_command(Path::new("/repository"));

    let mut command_arguments = Vec::new();
    for command_argument in swift_command.get_args() {
        command_arguments.push(command_argument.to_string_lossy().into_owned());
    }
    assert_eq!(swift_command.get_program(), "swift");
    assert_eq!(
        command_arguments,
        vec![
            "test",
            "--package-path",
            "/repository/apps/astronomical-menu",
            "-Xswiftc",
            "-warnings-as-errors"
        ]
    );
}

#[test]
fn should_clear_the_git_credential_helper_so_no_keychain_prompt_appears() {
    let swift_command = menu_contract_command(Path::new("/repository"));

    let mut credential_environment = Vec::new();
    for (variable_name, variable_value) in swift_command.get_envs() {
        let owned_value = variable_value
            .map(std::ffi::OsStr::to_string_lossy)
            .map(std::borrow::Cow::into_owned);
        credential_environment.push((variable_name.to_string_lossy().into_owned(), owned_value));
    }
    credential_environment.sort();
    assert_eq!(
        credential_environment,
        vec![
            ("GIT_CONFIG_COUNT".to_owned(), Some("1".to_owned())),
            (
                "GIT_CONFIG_KEY_0".to_owned(),
                Some("credential.helper".to_owned())
            ),
            ("GIT_CONFIG_VALUE_0".to_owned(), Some(String::new())),
        ]
    );
}

#[test]
fn should_refuse_arguments() {
    assert!(run(Path::new("/repository"), &["--fast".to_owned()]).is_err());
}
