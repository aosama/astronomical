use std::{
    ffi::OsString,
    io::Cursor,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use astronomical_cli::{
    LaunchArguments, LaunchDependencies, LaunchError, PreparedLaunch,
    opencode::OPENCODE_CONFIG_CONTENT_VARIABLE, prepare_launch,
};
use tempfile::TempDir;

use super::stub_server::{
    StubAstronomical, chat_models_json, embedding_model_json, image_model_json, mixed_library_json,
    ready_status_json,
};

const HTTP_TIMEOUT: Duration = Duration::from_secs(2);

fn install_fake_opencode(parent_directory: &Path) -> PathBuf {
    let program_path = parent_directory.join("opencode");
    std::fs::write(
        &program_path,
        "#!/bin/sh\nprintf '%s\\n' \"$OPENCODE_CONFIG_CONTENT\"\n",
    )
    .expect("fake OpenCode script");
    let mut permissions = std::fs::metadata(&program_path)
        .expect("fake OpenCode metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&program_path, permissions).expect("fake OpenCode executable bit");
    program_path
}

fn prepare_session(
    tool_slug: Option<&str>,
    model_id: Option<&str>,
    bind_address: std::net::SocketAddr,
    path_value: OsString,
    is_interactive: bool,
    stdin_text: &str,
) -> Result<(PreparedLaunch, String), LaunchError> {
    let mut stdin = Cursor::new(stdin_text.as_bytes().to_vec());
    let mut stderr_bytes = Vec::new();
    let prepared_launch = prepare_launch(
        LaunchArguments {
            tool_slug: tool_slug.map(str::to_owned),
            model_id: model_id.map(str::to_owned),
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![bind_address],
            path_value,
            is_interactive,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: HTTP_TIMEOUT,
        },
    )?;
    let stderr_text = String::from_utf8(stderr_bytes).expect("stderr utf8");
    Ok((prepared_launch, stderr_text))
}

fn config_from_prepared_launch(prepared_launch: &PreparedLaunch) -> serde_json::Value {
    let config_content = prepared_launch
        .extra_environment
        .iter()
        .find(|(name, _)| name == OPENCODE_CONFIG_CONTENT_VARIABLE)
        .map(|(_, value)| value.as_str())
        .expect("OPENCODE_CONFIG_CONTENT");
    serde_json::from_str(config_content).expect("OpenCode JSON")
}

#[test]
fn should_fail_when_opencode_is_missing() {
    let error = prepare_session(
        Some("opencode"),
        None,
        "127.0.0.1:1".parse().expect("unused port"),
        OsString::from("/tmp/astronomical-missing-opencode-path"),
        false,
        "",
    )
    .expect_err("missing OpenCode");
    assert!(matches!(error, LaunchError::OpenCodeMissing));
    assert_eq!(
        error.to_string(),
        "Install OpenCode: curl -fsSL https://opencode.ai/install | bash"
    );
}

#[test]
fn should_reject_an_unsupported_tool_name() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let error = prepare_session(
        Some("copilot"),
        None,
        "127.0.0.1:1".parse().expect("unused port"),
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("copilot is not in this slice");
    assert!(matches!(error, LaunchError::UnknownTool { .. }));
    assert_eq!(
        error.to_string(),
        "Unknown tool copilot. Try: astronomical launch opencode"
    );
}

#[test]
fn should_fail_when_astronomical_is_not_running() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let error = prepare_session(
        Some("opencode"),
        None,
        "127.0.0.1:1".parse().expect("unused port"),
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("daemon down");
    assert!(matches!(error, LaunchError::AstronomicalUnavailable));
    assert_eq!(error.to_string(), "Start Astronomical first.");
}

#[test]
fn should_attach_to_a_running_instance_when_the_preferred_channel_is_down() {
    // The developer case: an unpackaged binary prefers Development, but only the
    // Stable app is running. Launch must use Stable instead of refusing.
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let down_bind_address = "127.0.0.1:1".parse().expect("unused port");
    let mut stdin = Cursor::new(Vec::new());
    let mut stderr_bytes = Vec::new();
    let prepared_launch = prepare_launch(
        LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![down_bind_address, stub_server.bind_address],
            path_value: temporary_directory.path().as_os_str().to_os_string(),
            is_interactive: false,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: HTTP_TIMEOUT,
        },
    )
    .expect("falls back to the running instance");
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["provider"]["astronomical"]["options"]["baseURL"],
        format!("http://{}/v1", stub_server.bind_address)
    );
}

#[test]
fn should_prefer_the_first_healthy_candidate() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let preferred_stub = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("preferred-chat-model", 131_072)]),
    );
    let other_stub = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("other-chat-model", 131_072)]),
    );
    let mut stdin = Cursor::new(Vec::new());
    let mut stderr_bytes = Vec::new();
    let prepared_launch = prepare_launch(
        LaunchArguments {
            tool_slug: Some("opencode".to_owned()),
            model_id: None,
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![preferred_stub.bind_address, other_stub.bind_address],
            path_value: temporary_directory.path().as_os_str().to_os_string(),
            is_interactive: false,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: HTTP_TIMEOUT,
        },
    )
    .expect("preferred instance wins");
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["model"],
        "astronomical/preferred-chat-model"
    );
}

#[test]
fn should_fail_when_the_library_has_no_chat_models() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &image_model_json("library-image-model"),
    );
    let error = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("image-only library");
    assert!(matches!(error, LaunchError::NoChatModels));
    assert_eq!(
        error.to_string(),
        "No models in the Library yet. Open Astronomical and download one."
    );
}

#[test]
fn should_not_offer_embedding_models_as_launch_targets() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &embedding_model_json("library-embedding-model"),
    );
    let error = prepare_session(
        None,
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "1\n",
    )
    .expect_err("embeddings are not chat");
    assert!(matches!(error, LaunchError::NoChatModels));
}

#[test]
fn should_use_the_only_chat_model_without_prompting() {
    let temporary_directory = TempDir::new().expect("temp dir");
    let fake_opencode = install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let (prepared_launch, stderr_text) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "this stdin must be ignored\n",
    )
    .expect("single model launch");
    assert_eq!(prepared_launch.program, fake_opencode);
    assert!(!stderr_text.contains("Select a model"));
    let config_document = config_from_prepared_launch(&prepared_launch);
    assert_eq!(config_document["model"], "astronomical/library-chat-model");
    assert_eq!(
        config_document["provider"]["astronomical"]["options"]["baseURL"],
        format!("http://{}/v1", stub_server.bind_address)
    );
}

#[test]
fn should_treat_bare_launch_like_launch_opencode_when_opencode_is_installed() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let named = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("named");
    let bare = prepare_session(
        None,
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("bare");
    assert_eq!(
        config_from_prepared_launch(&named.0)["model"],
        config_from_prepared_launch(&bare.0)["model"]
    );
}

#[test]
fn should_skip_image_and_embedding_rows_when_one_chat_model_exists() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(ready_status_json(), &mixed_library_json());
    let (prepared_launch, stderr_text) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "",
    )
    .expect("mixed library still has one chat model");
    assert!(!stderr_text.contains("Select a model"));
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["model"],
        "astronomical/library-chat-model"
    );
}

#[test]
fn should_require_model_flag_when_several_chat_models_exist_without_a_terminal() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let error = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("non-interactive picker");
    assert!(matches!(error, LaunchError::ModelPickerRequired));
    assert_eq!(
        error.to_string(),
        "Choose a model with --model; several chat models are in the Library."
    );
}

#[test]
fn should_pick_a_chat_model_from_a_tty_list() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let (prepared_launch, stderr_text) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "2\n",
    )
    .expect("picker");
    assert!(stderr_text.contains("Select a model:"));
    assert!(stderr_text.contains("1. library-chat-model"));
    assert!(stderr_text.contains("2. second-library-chat-model"));
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["model"],
        "astronomical/second-library-chat-model"
    );
}

#[test]
fn should_accept_a_model_id_typed_into_the_picker() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let (prepared_launch, _) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "library-chat-model\n",
    )
    .expect("id pick");
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["model"],
        "astronomical/library-chat-model"
    );
}

#[test]
fn should_reject_an_invalid_picker_choice() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let error = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "0\n",
    )
    .expect_err("invalid picker");
    assert!(matches!(error, LaunchError::InvalidModelSelection));
}

#[test]
fn should_honor_model_flag_without_prompting() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let (prepared_launch, stderr_text) = prepare_session(
        Some("opencode"),
        Some("second-library-chat-model"),
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "1\n",
    )
    .expect("flag wins");
    assert!(!stderr_text.contains("Select a model"));
    assert_eq!(
        config_from_prepared_launch(&prepared_launch)["model"],
        "astronomical/second-library-chat-model"
    );
}

#[test]
fn should_reject_a_model_flag_that_is_not_a_library_chat_model() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let error = prepare_session(
        Some("opencode"),
        Some("missing-chat-model"),
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("unknown model");
    assert!(matches!(error, LaunchError::RequestedModelMissing { .. }));
}

#[test]
fn should_warn_when_the_selected_model_context_is_under_64k() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 32_768)]),
    );
    let (_, stderr_text) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("narrow context still launches");
    assert!(stderr_text.contains("64k or larger context window"));
    assert!(stderr_text.contains("32768"));
}

#[test]
fn should_overlay_opencode_config_in_the_child_environment_without_writing_user_files() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let home_directory = TempDir::new().expect("fake home");
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let (prepared_launch, _) = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("prepare");
    let command_output = Command::new(&prepared_launch.program)
        .env("HOME", home_directory.path())
        .env(
            OPENCODE_CONFIG_CONTENT_VARIABLE,
            prepared_launch
                .extra_environment
                .iter()
                .find(|(name, _)| name == OPENCODE_CONFIG_CONTENT_VARIABLE)
                .map(|(_, value)| value.as_str())
                .expect("config"),
        )
        .output()
        .expect("run fake OpenCode");
    assert!(command_output.status.success());
    let printed = String::from_utf8(command_output.stdout).expect("stdout");
    let config_document: serde_json::Value =
        serde_json::from_str(printed.trim()).expect("child JSON");
    assert_eq!(config_document["model"], "astronomical/library-chat-model");
    let user_opencode_config = home_directory.path().join(".config/opencode/opencode.json");
    assert!(
        !user_opencode_config.exists(),
        "launch must not write the user OpenCode config"
    );
}

#[test]
fn should_not_treat_a_json_listener_without_application_as_astronomical() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        r#"{"status":"ready"}"#,
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let error = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("non-Astronomical JSON is not a running instance");
    assert!(matches!(error, LaunchError::AstronomicalUnavailable));
}

#[test]
fn should_say_the_model_list_failed_when_status_is_healthy() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_opencode(temporary_directory.path());
    let stub_server = StubAstronomical::spawn_with_models_status(
        ready_status_json(),
        "{\"error\":\"no\"}",
        "500 Internal Server Error",
    );
    let error = prepare_session(
        Some("opencode"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("healthy status with a failed model list");
    assert!(matches!(error, LaunchError::ModelListUnavailable));
    assert_eq!(
        error.to_string(),
        "Astronomical is running but did not return a model list."
    );
}
