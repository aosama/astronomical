use std::{
    ffi::OsString,
    io::Cursor,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use astronomical_cli::{
    LaunchArguments, LaunchDependencies, LaunchError, PreparedLaunch, codex::CODEX_HOME_VARIABLE,
    prepare_launch,
};
use tempfile::TempDir;

use super::stub_server::{
    StubAstronomical, chat_models_json, embedding_model_json, image_model_json, ready_status_json,
};

const HTTP_TIMEOUT: Duration = Duration::from_secs(2);

/// The fake Codex prints the generated session configuration so tests assert
/// what the real Codex CLI would read from its CODEX_HOME.
fn install_fake_codex(parent_directory: &Path) -> PathBuf {
    let program_path = parent_directory.join("codex");
    std::fs::write(
        &program_path,
        "#!/bin/sh\ncat \"$CODEX_HOME/config.toml\"\n",
    )
    .expect("fake Codex script");
    let mut permissions = std::fs::metadata(&program_path)
        .expect("fake Codex metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&program_path, permissions).expect("fake Codex executable bit");
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

fn codex_home_path(prepared_launch: &PreparedLaunch) -> PathBuf {
    prepared_launch
        .extra_environment
        .iter()
        .find(|(name, _)| name == CODEX_HOME_VARIABLE)
        .map(|(_, value)| PathBuf::from(value))
        .expect("CODEX_HOME")
}

fn generated_config_text(prepared_launch: &PreparedLaunch) -> String {
    let codex_home_path = codex_home_path(prepared_launch);
    std::fs::read_to_string(codex_home_path.join("config.toml")).expect("generated config.toml")
}

#[test]
fn should_fail_when_codex_is_missing() {
    let error = prepare_session(
        Some("codex"),
        None,
        "127.0.0.1:1".parse().expect("unused port"),
        OsString::from("/tmp/astronomical-missing-codex-path"),
        false,
        "",
    )
    .expect_err("missing Codex");
    assert!(matches!(error, LaunchError::CodexMissing));
    assert_eq!(
        error.to_string(),
        "Install Codex CLI: npm install -g @openai/codex"
    );
}

#[test]
fn should_reject_an_unknown_tool_by_listing_the_supported_tools() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
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
        "Unknown tool copilot. Supported tools: opencode, codex"
    );
}

#[test]
fn should_start_codex_on_the_only_chat_model_without_prompting() {
    let temporary_directory = TempDir::new().expect("temp dir");
    let fake_codex = install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let (prepared_launch, stderr_text) = prepare_session(
        Some("codex"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "this stdin must be ignored\n",
    )
    .expect("single model launch");
    assert_eq!(prepared_launch.program, fake_codex);
    assert!(!stderr_text.contains("Select a model"));
    let config_text = generated_config_text(&prepared_launch);
    assert!(
        config_text.contains("model = \"library-chat-model\""),
        "selected chat model must be the session model: {config_text}"
    );
    assert!(
        config_text.contains("model_provider = \"astronomical\""),
        "session must target the Astronomical provider: {config_text}"
    );
    assert!(
        config_text.contains(&format!(
            "base_url = \"http://{}/v1\"",
            stub_server.bind_address
        )),
        "session must target the OpenAI-compatible local endpoint: {config_text}"
    );
    assert!(
        config_text.contains("wire_api = \"responses\""),
        "Codex speaks the responses wire protocol: {config_text}"
    );
}

#[test]
fn should_leave_the_users_codex_configuration_untouched() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let home_directory = TempDir::new().expect("fake home");
    let user_codex_home = home_directory.path().join(".codex");
    std::fs::create_dir_all(&user_codex_home).expect("user .codex");
    let user_config_path = user_codex_home.join("config.toml");
    std::fs::write(&user_config_path, "# the user's own configuration\n").expect("user config");
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let (prepared_launch, _) = prepare_session(
        Some("codex"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("prepare");
    let generated_codex_home = codex_home_path(&prepared_launch);
    assert_ne!(
        generated_codex_home, user_codex_home,
        "the session CODEX_HOME must not be the user's .codex"
    );
    let command_output = Command::new(&prepared_launch.program)
        .env("HOME", home_directory.path())
        .env(CODEX_HOME_VARIABLE, &generated_codex_home)
        .output()
        .expect("run fake Codex");
    assert!(command_output.status.success());
    let printed = String::from_utf8(command_output.stdout).expect("stdout");
    assert!(printed.contains("model = \"library-chat-model\""));
    assert_eq!(
        std::fs::read_to_string(&user_config_path).expect("user config after launch"),
        "# the user's own configuration\n",
        "launch must not rewrite the user Codex config"
    );
}

#[test]
fn should_require_the_model_flag_for_codex_when_several_chat_models_exist_without_a_terminal() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let error = prepare_session(
        Some("codex"),
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
fn should_word_the_narrow_context_advisory_for_codex() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 32_768)]),
    );
    let (_, stderr_text) = prepare_session(
        Some("codex"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("narrow context still launches");
    assert!(stderr_text.contains("Codex works better with a 64k or larger context window"));
    assert!(stderr_text.contains("32768"));
}

#[test]
fn should_not_offer_image_or_embedding_advertisements_to_codex() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let image_only_server = StubAstronomical::spawn(
        ready_status_json(),
        &image_model_json("library-image-model"),
    );
    let image_error = prepare_session(
        Some("codex"),
        None,
        image_only_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "1\n",
    )
    .expect_err("images are not chat");
    assert!(matches!(image_error, LaunchError::NoChatModels));
    let embedding_only_server = StubAstronomical::spawn(
        ready_status_json(),
        &embedding_model_json("library-embedding-model"),
    );
    let embedding_error = prepare_session(
        Some("codex"),
        None,
        embedding_only_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "1\n",
    )
    .expect_err("embeddings are not chat");
    assert!(matches!(embedding_error, LaunchError::NoChatModels));
}

#[test]
fn should_generate_the_exact_session_config_document() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let (prepared_launch, _) = prepare_session(
        Some("codex"),
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect("prepare");
    let expected_config = format!(
        "model = \"library-chat-model\"\n\
         model_provider = \"astronomical\"\n\
         \n\
         [model_providers.astronomical]\n\
         name = \"Astronomical\"\n\
         base_url = \"http://{}/v1\"\n\
         wire_api = \"responses\"\n",
        stub_server.bind_address
    );
    assert_eq!(generated_config_text(&prepared_launch), expected_config);
}

#[test]
fn should_keep_the_default_harness_opencode_when_only_codex_is_installed() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[("library-chat-model", 131_072)]),
    );
    let error = prepare_session(
        None,
        None,
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        false,
        "",
    )
    .expect_err("bare launch still means OpenCode");
    assert!(matches!(error, LaunchError::OpenCodeMissing));
}

#[test]
fn should_honor_the_model_flag_for_codex_without_prompting() {
    let temporary_directory = TempDir::new().expect("temp dir");
    install_fake_codex(temporary_directory.path());
    let stub_server = StubAstronomical::spawn(
        ready_status_json(),
        &chat_models_json(&[
            ("library-chat-model", 131_072),
            ("second-library-chat-model", 32_768),
        ]),
    );
    let (prepared_launch, stderr_text) = prepare_session(
        Some("codex"),
        Some("second-library-chat-model"),
        stub_server.bind_address,
        temporary_directory.path().as_os_str().to_os_string(),
        true,
        "1\n",
    )
    .expect("flag wins");
    assert!(!stderr_text.contains("Select a model"));
    assert!(
        generated_config_text(&prepared_launch).contains("model = \"second-library-chat-model\""),
        "the --model flag must select the session model"
    );
}
