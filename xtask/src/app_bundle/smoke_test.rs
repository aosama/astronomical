//! `cargo xtask app smoke-test --app-bundle PATH [--real-model MODEL_ID]`:
//! launches a built bundle's daemon and menu-bar agent in isolation, checks that
//! they serve this build's identity, and shuts them down. It then starts the
//! agent the way macOS does at login and requires the agent to bring the
//! engine up by itself (`login_launch_check`). Typed port of the launching
//! half of the retired `validate-macos-app.sh`.
//!
//! It never stops an instance it did not start: an occupied port fails the
//! run. `--real-model` sends one Romeo and Juliet prompt through a model the
//! instance advertises; shared GPU memory is not isolated, so it is opt-in.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::channel_profile::{AGENT_EXECUTABLE_NAME, Channel, ChannelProfile, SandboxPolicy};
use super::login_launch_check::check_login_launch_starts_engine;
use super::tool_command::run_quiet_tool;

pub(super) const HEALTH_WAIT: Duration = Duration::from_secs(30);
const AGENT_SURVIVAL_PERIOD: Duration = Duration::from_secs(2);
const PROCESS_EXIT_WAIT: Duration = Duration::from_secs(30);
pub(super) const POLL_INTERVAL: Duration = Duration::from_millis(250);
pub(super) const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const ROMEO_FIXTURE_PATH: &str =
    "verification/acceptance/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SmokeTestRequest {
    pub(crate) bundle_path: PathBuf,
    pub(crate) real_model_identifier: Option<String>,
}

/// What the bundle's plist says the running instance must report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundleInstanceIdentity {
    pub(crate) channel_name: String,
    /// The engine instance the channel runs, which the engine reports as its
    /// channel: `stable` for a store bundle, whose channel is `app-store`.
    pub(crate) runtime_instance_name: String,
    /// A sandboxed bundle's engine can be started only by its agent.
    pub(crate) is_sandboxed: bool,
    pub(crate) daemon_port: u64,
    pub(crate) application_version: String,
    pub(crate) build_commit: String,
}

pub(crate) fn run_smoke_test(
    repository_root: &Path,
    smoke_test_request: &SmokeTestRequest,
) -> Result<(), String> {
    let instance_identity = read_bundle_instance_identity(&smoke_test_request.bundle_path)?;
    let daemon_base_url = format!("http://127.0.0.1:{}", instance_identity.daemon_port);
    if http_get(&format!("{daemon_base_url}/health")).is_ok() {
        return Err(format!(
            "the {} endpoint is already occupied; the smoke test will not stop an instance it did not start",
            instance_identity.channel_name
        ));
    }
    let log_directory = smoke_test_log_directory()?;
    if instance_identity.is_sandboxed {
        // The engine of a sandboxed bundle is signed to inherit its sandbox
        // from the agent that starts it. Started directly it would run
        // outside any sandbox and write state where the real build never
        // does, so only the agent-driven launch below is exercised.
        println!(
            "[app-smoke-test] step=daemon status=skipped reason=sandboxed-engine-starts-only-from-its-agent"
        );
    } else {
        check_directly_launched_instance(
            repository_root,
            smoke_test_request,
            &instance_identity,
            &daemon_base_url,
            &log_directory,
        )?;
    }
    // Runs only once the directly launched daemon has exited: the agent
    // starts an engine only when the channel's endpoint is unoccupied.
    check_login_launch_starts_engine(
        &smoke_test_request.bundle_path,
        &instance_identity,
        &daemon_base_url,
        &log_directory,
    )?;
    println!(
        "[app-smoke-test] validated {} {} ({}) without replacing another instance",
        instance_identity.application_version,
        instance_identity.channel_name,
        instance_identity.build_commit
    );
    Ok(())
}

/// Starts the bundle's daemon itself, exercises it and the agent beside it,
/// and shuts the daemon down whether or not the exercise passed.
fn check_directly_launched_instance(
    repository_root: &Path,
    smoke_test_request: &SmokeTestRequest,
    instance_identity: &BundleInstanceIdentity,
    daemon_base_url: &str,
    log_directory: &Path,
) -> Result<(), String> {
    let mut daemon_process = launch_bundled_executable(
        &smoke_test_request
            .bundle_path
            .join("Contents/MacOS/astronomicald"),
        &["--instance", &instance_identity.runtime_instance_name],
        &log_directory.join("daemon.log"),
    )?;
    let smoke_result = exercise_running_instance(
        repository_root,
        smoke_test_request,
        instance_identity,
        daemon_base_url,
        log_directory,
    );
    let shutdown_result = shut_down_daemon(&mut daemon_process, daemon_base_url);
    smoke_result?;
    shutdown_result
}

fn exercise_running_instance(
    repository_root: &Path,
    smoke_test_request: &SmokeTestRequest,
    instance_identity: &BundleInstanceIdentity,
    daemon_base_url: &str,
    log_directory: &Path,
) -> Result<(), String> {
    wait_for_health(daemon_base_url, log_directory)?;
    let status_document = http_get_json(&format!("{daemon_base_url}/v1/status"))?;
    if let Some(identity_problem) = status_identity_problem(&status_document, instance_identity) {
        return Err(identity_problem);
    }
    let models_document = http_get_json(&format!("{daemon_base_url}/v1/models"))?;
    if !models_document["data"].is_array() {
        return Err("/v1/models did not return a data array".to_owned());
    }
    println!("[app-smoke-test] step=daemon status=success");
    check_agent_survives(&smoke_test_request.bundle_path, log_directory)?;
    if let Some(real_model_identifier) = &smoke_test_request.real_model_identifier {
        probe_real_model(
            repository_root,
            daemon_base_url,
            &models_document,
            real_model_identifier,
        )?;
    }
    Ok(())
}

/// The running daemon must be the one in this bundle, not a stale build.
pub(crate) fn status_identity_problem(
    status_document: &serde_json::Value,
    instance_identity: &BundleInstanceIdentity,
) -> Option<String> {
    let application_document = &status_document["application"];
    let reports_bundle_identity = application_document["channel"].as_str()
        == Some(instance_identity.runtime_instance_name.as_str())
        && application_document["version"].as_str()
            == Some(instance_identity.application_version.as_str())
        && application_document["commit"].as_str() == Some(instance_identity.build_commit.as_str());
    if reports_bundle_identity {
        return None;
    }
    Some(format!(
        "the running daemon reports {application_document}, not {} {} ({})",
        instance_identity.channel_name,
        instance_identity.application_version,
        instance_identity.build_commit
    ))
}

pub(crate) fn instance_identity_from_plist(
    plist_document: &serde_json::Value,
) -> Result<BundleInstanceIdentity, String> {
    let (Some(channel_name), Some(daemon_port), Some(application_version), Some(build_commit)) = (
        plist_document["AstronomicalChannel"].as_str(),
        plist_document["AstronomicalDaemonPort"].as_u64(),
        plist_document["CFBundleShortVersionString"].as_str(),
        plist_document["AstronomicalBuildCommit"].as_str(),
    ) else {
        return Err("the bundle plist lacks its channel, port, version, or commit".to_owned());
    };
    bundle_instance_identity(channel_name, daemon_port, application_version, build_commit)
}

/// The identity a bundle of `channel_name` must report, with what the
/// channel implies: the engine instance it runs and whether it is sandboxed.
/// An unknown channel name is refused, since nothing is known about it.
fn bundle_instance_identity(
    channel_name: &str,
    daemon_port: u64,
    application_version: &str,
    build_commit: &str,
) -> Result<BundleInstanceIdentity, String> {
    let channel = Channel::from_channel_name(channel_name)?;
    Ok(BundleInstanceIdentity {
        channel_name: channel_name.to_owned(),
        runtime_instance_name: channel.runtime_instance_name().to_owned(),
        is_sandboxed: ChannelProfile::for_channel(channel).sandbox_policy
            == SandboxPolicy::AppStoreSandbox,
        daemon_port,
        application_version: application_version.to_owned(),
        build_commit: build_commit.to_owned(),
    })
}

fn read_bundle_instance_identity(bundle_path: &Path) -> Result<BundleInstanceIdentity, String> {
    if !bundle_path.is_dir() {
        return Err(format!("app bundle not found: {}", bundle_path.display()));
    }
    instance_identity_from_plist(&read_plist_document(
        &bundle_path.join("Contents/Info.plist"),
    )?)
}

pub(super) fn read_plist_document(plist_path: &Path) -> Result<serde_json::Value, String> {
    let plist_json_text = run_quiet_tool(
        Command::new("plutil")
            .args(["-convert", "json", "-o", "-"])
            .arg(plist_path),
    )?;
    match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => Ok(plist_document),
        Err(parse_error) => Err(format!(
            "{} did not convert to JSON: {parse_error}",
            plist_path.display()
        )),
    }
}

/// The menu-bar agent must start and stay up; it is then asked to terminate.
/// The main app is not launched here: it would start an agent of its own
/// through LaunchServices, outside this test's control.
fn check_agent_survives(bundle_path: &Path, log_directory: &Path) -> Result<(), String> {
    let mut agent_process = launch_bundled_executable(
        &agent_executable_path(bundle_path)?,
        &[],
        &log_directory.join("agent.log"),
    )?;
    std::thread::sleep(AGENT_SURVIVAL_PERIOD);
    let agent_survived = matches!(agent_process.try_wait(), Ok(None));
    let _ = Command::new("kill")
        .args(["-TERM", &agent_process.id().to_string()])
        .status();
    wait_for_child_exit(&mut agent_process, "menu-bar agent")?;
    if !agent_survived {
        return Err(format!(
            "the menu-bar agent exited during the smoke test; read {}",
            log_directory.join("agent.log").display()
        ));
    }
    println!("[app-smoke-test] step=menu status=success");
    Ok(())
}

/// The agent's bundle name differs per channel, and the smoke test is handed
/// only a bundle path, so the one bundle in LoginItems is found by looking.
pub(crate) fn agent_executable_path(bundle_path: &Path) -> Result<PathBuf, String> {
    let login_items_directory = bundle_path.join("Contents/Library/LoginItems");
    let Ok(login_item_entries) = std::fs::read_dir(&login_items_directory) else {
        return Err(format!(
            "the bundle has no menu-bar agent: {} is missing",
            login_items_directory.display()
        ));
    };
    for login_item_entry in login_item_entries.flatten() {
        let agent_executable = login_item_entry
            .path()
            .join("Contents/MacOS")
            .join(AGENT_EXECUTABLE_NAME);
        if agent_executable.is_file() {
            return Ok(agent_executable);
        }
    }
    Err(format!(
        "no menu-bar agent executable under {}",
        login_items_directory.display()
    ))
}

fn probe_real_model(
    repository_root: &Path,
    daemon_base_url: &str,
    models_document: &serde_json::Value,
    real_model_identifier: &str,
) -> Result<(), String> {
    println!(
        "[app-smoke-test] shared GPU and wired memory are not isolated; this probe may affect other instances' latency"
    );
    if !advertises_model(models_document, real_model_identifier) {
        return Err(format!(
            "the probe model is not advertised: {real_model_identifier}"
        ));
    }
    let romeo_prompt = match std::fs::read_to_string(repository_root.join(ROMEO_FIXTURE_PATH)) {
        Ok(romeo_prompt) => romeo_prompt,
        Err(read_error) => {
            return Err(format!(
                "could not read the Romeo and Juliet fixture: {read_error}"
            ));
        }
    };
    let chat_request = serde_json::json!({
        "model": real_model_identifier,
        "messages": [{"role": "user", "content": romeo_prompt}],
        "max_tokens": 16,
        "stream": false
    });
    let chat_response = http_post_json(
        &format!("{daemon_base_url}/v1/chat/completions"),
        &chat_request.to_string(),
    )?;
    if !has_assistant_output(&chat_response) {
        return Err(format!(
            "the probe model returned no assistant text or reasoning: {real_model_identifier}"
        ));
    }
    println!("[app-smoke-test] step=real-model model={real_model_identifier} status=success");
    Ok(())
}

pub(crate) fn advertises_model(
    models_document: &serde_json::Value,
    model_identifier: &str,
) -> bool {
    let Some(advertised_models) = models_document["data"].as_array() else {
        return false;
    };
    for advertised_model in advertised_models {
        if advertised_model["id"].as_str() == Some(model_identifier) {
            return true;
        }
    }
    false
}

pub(crate) fn has_assistant_output(chat_response: &serde_json::Value) -> bool {
    let assistant_message = &chat_response["choices"][0]["message"];
    for output_field in ["content", "reasoning_content"] {
        if let Some(output_text) = assistant_message[output_field].as_str()
            && !output_text.is_empty()
        {
            return true;
        }
    }
    false
}

fn shut_down_daemon(daemon_process: &mut Child, daemon_base_url: &str) -> Result<(), String> {
    let _ = http_post_json(&format!("{daemon_base_url}/v1/control/shutdown"), "{}");
    wait_for_child_exit(daemon_process, "daemon")?;
    println!("[app-smoke-test] step=shutdown status=success");
    Ok(())
}

fn wait_for_health(daemon_base_url: &str, log_directory: &Path) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while http_get(&format!("{daemon_base_url}/health")).is_err() {
        if wait_started_at.elapsed() >= HEALTH_WAIT {
            return Err(format!(
                "the daemon did not become healthy within {}s; read {}",
                HEALTH_WAIT.as_secs(),
                log_directory.join("daemon.log").display()
            ));
        }
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            println!(
                "[app-smoke-test] waiting for health: {:.0}s of {}s",
                wait_started_at.elapsed().as_secs_f64(),
                HEALTH_WAIT.as_secs()
            );
            last_progress_at = Instant::now();
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    Ok(())
}

pub(super) fn wait_for_child_exit(
    child_process: &mut Child,
    process_description: &str,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    loop {
        match child_process.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if wait_started_at.elapsed() < PROCESS_EXIT_WAIT => {
                std::thread::sleep(POLL_INTERVAL)
            }
            Ok(None) => {
                // The smoke test started this process, so it may stop it.
                let _ = child_process.kill();
                return Err(format!(
                    "the {process_description} did not exit within {}s and was stopped",
                    PROCESS_EXIT_WAIT.as_secs()
                ));
            }
            Err(wait_error) => {
                return Err(format!(
                    "could not wait for the {process_description}: {wait_error}"
                ));
            }
        }
    }
}

fn launch_bundled_executable(
    executable_path: &Path,
    launch_arguments: &[&str],
    log_path: &Path,
) -> Result<Child, String> {
    let mut launch_command = Command::new(executable_path);
    launch_command.args(launch_arguments);
    spawn_with_logs(&mut launch_command, log_path)
}

/// Starts `launch_command` detached from this terminal, with its output kept
/// in `log_path` and its sibling `.stderr.log` for the failure message.
pub(super) fn spawn_with_logs(
    launch_command: &mut Command,
    log_path: &Path,
) -> Result<Child, String> {
    let (Ok(standard_output_log), Ok(standard_error_log)) = (
        File::create(log_path),
        File::create(log_path.with_extension("stderr.log")),
    ) else {
        return Err(format!("could not create {}", log_path.display()));
    };
    match launch_command
        .stdin(Stdio::null())
        .stdout(standard_output_log)
        .stderr(standard_error_log)
        .spawn()
    {
        Ok(launched_process) => Ok(launched_process),
        Err(spawn_error) => Err(format!(
            "could not launch {}: {spawn_error}",
            launch_command.get_program().to_string_lossy()
        )),
    }
}

fn smoke_test_log_directory() -> Result<PathBuf, String> {
    let log_directory =
        std::env::temp_dir().join(format!("astronomical-smoke-test.{}", std::process::id()));
    match std::fs::create_dir_all(&log_directory) {
        Ok(()) => Ok(log_directory),
        Err(create_error) => Err(format!(
            "could not create {}: {create_error}",
            log_directory.display()
        )),
    }
}

pub(super) fn http_get(request_url: &str) -> Result<String, String> {
    run_quiet_tool(Command::new("curl").args([
        "--silent",
        "--show-error",
        "--fail",
        "--max-time",
        "10",
        request_url,
    ]))
}

pub(super) fn http_get_json(request_url: &str) -> Result<serde_json::Value, String> {
    parse_json_response(request_url, &http_get(request_url)?)
}

/// A real-model probe can take a while on a cold model, hence two minutes.
pub(super) fn http_post_json(
    request_url: &str,
    request_body: &str,
) -> Result<serde_json::Value, String> {
    let response_text = run_quiet_tool(Command::new("curl").args([
        "--silent",
        "--show-error",
        "--fail",
        "--max-time",
        "120",
        "--header",
        "Content-Type: application/json",
        "--data-binary",
        request_body,
        request_url,
    ]))?;
    if response_text.is_empty() {
        return Ok(serde_json::Value::Null);
    }
    parse_json_response(request_url, &response_text)
}

fn parse_json_response(
    request_url: &str,
    response_text: &str,
) -> Result<serde_json::Value, String> {
    match serde_json::from_str(response_text) {
        Ok(response_document) => Ok(response_document),
        Err(parse_error) => Err(format!("{request_url} did not return JSON: {parse_error}")),
    }
}
