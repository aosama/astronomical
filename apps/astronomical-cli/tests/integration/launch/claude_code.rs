use std::{
    ffi::OsString,
    io::Cursor,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use astronomical_cli::{
    LaunchArguments, LaunchDependencies, LaunchError, PreparedLaunch, prepare_launch,
    verbs::launch::claude_code_config::{
        ANTHROPIC_BASE_URL_VARIABLE, ANTHROPIC_MODEL_VARIABLE,
        CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE,
    },
};
use tempfile::TempDir;

use crate::integration::support::stub_loopback_http::{
    StubAstronomical, chat_models_json, mixed_library_json, ready_status_json,
};

const HTTP_TIMEOUT: Duration = Duration::from_secs(2);

/// The fake Claude Code prints the environment and arguments it was started
/// with, so a journey asserts what the real Claude Code would read.
fn install_fake_claude_code(parent_directory: &Path) -> PathBuf {
    let program_path = parent_directory.join("claude");
    std::fs::write(
        &program_path,
        "#!/bin/sh\necho \"base=$ANTHROPIC_BASE_URL model=$ANTHROPIC_MODEL window=$CLAUDE_CODE_MAX_CONTEXT_TOKENS arguments=$*\"\n",
    )
    .expect("fake Claude Code script");
    let mut permissions = std::fs::metadata(&program_path)
        .expect("fake Claude Code metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&program_path, permissions).expect("fake Claude Code executable bit");
    program_path
}

fn prepare_claude_session(
    bind_address: std::net::SocketAddr,
    path_value: OsString,
    harness_arguments: Vec<OsString>,
) -> Result<PreparedLaunch, LaunchError> {
    let mut stdin = Cursor::new(Vec::new());
    let mut stderr_bytes = Vec::new();
    prepare_launch(
        LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: None,
            harness_arguments,
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![bind_address],
            path_value,
            is_interactive: false,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: HTTP_TIMEOUT,
        },
    )
}

fn environment_value<'a>(
    prepared_launch: &'a PreparedLaunch,
    variable_name: &str,
) -> Option<&'a str> {
    for (environment_name, environment_value) in &prepared_launch.extra_environment {
        if environment_name == variable_name {
            return Some(environment_value);
        }
    }
    None
}

#[test]
fn should_fail_with_the_install_hint_when_claude_code_is_missing() {
    let launch_error = prepare_claude_session(
        "127.0.0.1:1".parse().expect("unused port"),
        OsString::from("/tmp/astronomical-missing-claude-path"),
        Vec::new(),
    )
    .expect_err("missing Claude Code");

    assert!(matches!(launch_error, LaunchError::ClaudeCodeMissing));
    assert_eq!(
        launch_error.to_string(),
        "Install Claude Code: curl -fsSL https://claude.ai/install.sh | bash"
    );
}

#[test]
fn should_refuse_when_astronomical_is_not_running() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_claude_code(temporary_directory.path());

    let launch_error = prepare_claude_session(
        "127.0.0.1:1".parse().expect("unused port"),
        temporary_directory.path().as_os_str().to_os_string(),
        Vec::new(),
    )
    .expect_err("no running daemon");

    assert!(matches!(launch_error, LaunchError::AstronomicalUnavailable));
}

#[test]
fn should_start_claude_code_on_the_local_endpoint_with_the_models_window() {
    let temporary_directory = TempDir::new().expect("temp dir");
    let fake_claude_code = install_fake_claude_code(temporary_directory.path());
    // An image, a chat, and an embedding model: only the chat model can drive
    // a coding harness, so the launch must pick it without asking.
    let stub_server = StubAstronomical::spawn(ready_status_json(), &mixed_library_json());

    let prepared_launch = prepare_claude_session(
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        Vec::new(),
    )
    .expect("Claude Code launch");

    assert_eq!(prepared_launch.program, fake_claude_code);
    assert_eq!(
        environment_value(&prepared_launch, ANTHROPIC_BASE_URL_VARIABLE),
        Some(format!("http://{}", stub_server.bind_address).as_str())
    );
    assert_eq!(
        environment_value(&prepared_launch, ANTHROPIC_MODEL_VARIABLE),
        Some("library-chat-model"),
        "the image model is not a Claude Code launch target"
    );
    assert_eq!(
        environment_value(&prepared_launch, CLAUDE_CODE_MAX_CONTEXT_TOKENS_VARIABLE),
        Some("131072")
    );
}

#[test]
fn should_hand_harness_arguments_and_environment_to_the_started_claude_code() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_claude_code(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 262_144)]),
    );
    let prepared_launch = prepare_claude_session(
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        vec!["-p".into(), "Shall I compare thee?".into()],
    )
    .expect("Claude Code launch");

    let mut fake_claude_command = Command::new(&prepared_launch.program);
    fake_claude_command.args(&prepared_launch.arguments);
    for (environment_name, environment_value) in &prepared_launch.extra_environment {
        fake_claude_command.env(environment_name, environment_value);
    }
    let command_output = fake_claude_command.output().expect("run fake Claude Code");

    assert!(command_output.status.success());
    let printed = String::from_utf8(command_output.stdout).expect("stdout");
    assert!(printed.contains("model=library-chat-model"), "{printed}");
    assert!(printed.contains("window=262144"), "{printed}");
    assert!(
        printed.contains("arguments=-p Shall I compare thee?"),
        "{printed}"
    );
}
