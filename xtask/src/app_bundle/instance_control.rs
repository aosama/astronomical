//! `cargo xtask app instance status|start|stop|restart` for the Development
//! instance, the typed replacement for the retired shell instance script.
//!
//! `start` runs the bundled daemon headless from the caller's shell, so
//! exported `ASTRONOMICAL_*` experiment flags reach it; opening the app
//! through LaunchServices would drop them. `stop` quits the menu app first,
//! because a running menu app restarts a daemon it sees disappear. `restart`
//! reopens a menu app it quit, once the daemon is healthy, so the menu app
//! attaches to that daemon and the person keeps their menu bar icon.

use std::fs::File;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::channel_profile::{Channel, ChannelProfile};
use super::dev_pipeline::DevPipelinePaths;
use super::running_instance::{open_bundle, quit_menu_app, wait_for_daemon_exit};

const HEALTH_WAIT: Duration = Duration::from_secs(30);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(1);
const HEALTH_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstanceAction {
    Status,
    Start,
    Stop,
    Restart,
}

impl InstanceAction {
    pub(crate) fn from_action_name(action_name: &str) -> Result<Self, String> {
        match action_name {
            "status" => Ok(Self::Status),
            "start" => Ok(Self::Start),
            "stop" => Ok(Self::Stop),
            "restart" => Ok(Self::Restart),
            _ => Err(format!(
                "instance action must be status, start, stop, or restart (got {action_name:?})"
            )),
        }
    }
}

pub(crate) fn run_instance_action(
    repository_root: &Path,
    instance_action: InstanceAction,
) -> Result<(), String> {
    let channel_profile = ChannelProfile::for_channel(Channel::Development);
    let bundle_path = DevPipelinePaths::for_profile(repository_root, &channel_profile).bundle_path;
    match instance_action {
        InstanceAction::Status => print_status(repository_root, &channel_profile),
        InstanceAction::Start => start_instance(&bundle_path, &channel_profile),
        InstanceAction::Stop => stop_instance(&channel_profile).map(drop),
        InstanceAction::Restart => {
            restart_instance(&bundle_path, &channel_profile)?;
            print_status(repository_root, &channel_profile)
        }
    }
}

fn restart_instance(bundle_path: &Path, channel_profile: &ChannelProfile) -> Result<(), String> {
    let was_menu_app_running = stop_instance(channel_profile)?;
    start_instance(bundle_path, channel_profile)?;
    reopen_menu_app_if_it_was_running(was_menu_app_running, bundle_path, open_bundle)
}

/// Returns whether a menu app was running, which `restart` needs to put it back.
fn stop_instance(channel_profile: &ChannelProfile) -> Result<bool, String> {
    let was_menu_app_running = quit_menu_app(channel_profile)?;
    wait_for_daemon_exit(channel_profile)?;
    println!("[app-instance] {} is stopped", channel_profile.bundle_name);
    Ok(was_menu_app_running)
}

/// Runs after the daemon is healthy, because a menu app that opens first would
/// start its own daemon, without the caller's exported flags. A headless session
/// stays headless: a menu app that was not running is not opened.
pub(crate) fn reopen_menu_app_if_it_was_running(
    was_menu_app_running: bool,
    bundle_path: &Path,
    open_menu_app: fn(&Path) -> Result<(), String>,
) -> Result<(), String> {
    if !was_menu_app_running {
        return Ok(());
    }
    match open_menu_app(bundle_path) {
        Ok(()) => {
            println!("[app-instance] reopened the menu app");
            Ok(())
        }
        Err(open_error) => Err(format!(
            "the daemon is running, but the menu app did not reopen: {open_error}"
        )),
    }
}

fn start_instance(bundle_path: &Path, channel_profile: &ChannelProfile) -> Result<(), String> {
    let daemon_executable = bundle_path.join("Contents/MacOS/astronomicald");
    if !daemon_executable.is_file() {
        return Err(format!(
            "the {} bundle is missing at {}; build it with: cargo xtask app dev",
            channel_profile.bundle_name,
            bundle_path.display()
        ));
    }
    if daemon_is_healthy(channel_profile) {
        return Err("the instance is already running; use restart or stop first".to_owned());
    }
    let log_directory = state_directory(channel_profile)?.join("logs");
    if let Err(create_error) = std::fs::create_dir_all(&log_directory) {
        return Err(format!(
            "could not create {}: {create_error}",
            log_directory.display()
        ));
    }
    let console_log_path = log_directory.join("daemon-console.log");
    let daemon_process_id =
        spawn_headless_daemon(&daemon_executable, channel_profile, &console_log_path)?;
    wait_for_health(channel_profile, &console_log_path)?;
    println!(
        "[app-instance] {} is up on http://127.0.0.1:{} (daemon pid {})\n[app-instance] logs: {}",
        channel_profile.bundle_name,
        channel_profile.daemon_port,
        daemon_process_id,
        log_directory.display()
    );
    Ok(())
}

/// Its own process group keeps the daemon alive after this command and the
/// terminal that ran it exit, as the shell script's `nohup` did.
fn spawn_headless_daemon(
    daemon_executable: &Path,
    channel_profile: &ChannelProfile,
    console_log_path: &Path,
) -> Result<u32, String> {
    let console_log = open_console_log(console_log_path)?;
    let console_log_for_errors = clone_log_handle(&console_log, console_log_path)?;
    let spawn_result = Command::new(daemon_executable)
        .args(["--instance", channel_profile.channel.as_channel_name()])
        .stdin(Stdio::null())
        .stdout(console_log)
        .stderr(console_log_for_errors)
        .process_group(0)
        .spawn();
    match spawn_result {
        Ok(daemon_process) => Ok(daemon_process.id()),
        Err(spawn_error) => Err(format!("could not start the daemon: {spawn_error}")),
    }
}

fn open_console_log(console_log_path: &Path) -> Result<File, String> {
    match File::create(console_log_path) {
        Ok(console_log) => Ok(console_log),
        Err(create_error) => Err(format!(
            "could not open {}: {create_error}",
            console_log_path.display()
        )),
    }
}

fn clone_log_handle(console_log: &File, console_log_path: &Path) -> Result<File, String> {
    match console_log.try_clone() {
        Ok(cloned_log) => Ok(cloned_log),
        Err(clone_error) => Err(format!(
            "could not share {}: {clone_error}",
            console_log_path.display()
        )),
    }
}

fn wait_for_health(
    channel_profile: &ChannelProfile,
    console_log_path: &Path,
) -> Result<(), String> {
    let wait_started_at = Instant::now();
    let mut last_progress_at = Instant::now();
    while !daemon_is_healthy(channel_profile) {
        if wait_started_at.elapsed() >= HEALTH_WAIT {
            return Err(format!(
                "the daemon did not become healthy within {}s; read {}",
                HEALTH_WAIT.as_secs(),
                console_log_path.display()
            ));
        }
        if last_progress_at.elapsed() >= HEALTH_PROGRESS_INTERVAL {
            println!(
                "[app-instance] waiting for health: {:.0}s of {}s",
                wait_started_at.elapsed().as_secs_f64(),
                HEALTH_WAIT.as_secs()
            );
            last_progress_at = Instant::now();
        }
        std::thread::sleep(HEALTH_POLL_INTERVAL);
    }
    Ok(())
}

/// `/health` is the daemon's liveness route; `/v1/health` answers 404. The
/// retired shell script probed the latter without `--fail`, so a 404 read
/// as healthy and its start never really waited for the daemon.
fn daemon_is_healthy(channel_profile: &ChannelProfile) -> bool {
    match Command::new("curl")
        .args(["--silent", "--fail", "--max-time", "2"])
        .arg(format!(
            "http://127.0.0.1:{}/health",
            channel_profile.daemon_port
        ))
        .output()
    {
        Ok(curl_output) => curl_output.status.success(),
        Err(_) => false,
    }
}

fn print_status(repository_root: &Path, channel_profile: &ChannelProfile) -> Result<(), String> {
    let status_output = Command::new("curl")
        .args(["--silent", "--fail", "--max-time", "3"])
        .arg(format!(
            "http://127.0.0.1:{}/v1/status",
            channel_profile.daemon_port
        ))
        .output();
    let status_text = match status_output {
        Ok(curl_output) if curl_output.status.success() => {
            String::from_utf8_lossy(&curl_output.stdout).into_owned()
        }
        _ => {
            println!(
                "{}: not responding on http://127.0.0.1:{}",
                channel_profile.bundle_name, channel_profile.daemon_port
            );
            return Ok(());
        }
    };
    let status_document: serde_json::Value = match serde_json::from_str(&status_text) {
        Ok(status_document) => status_document,
        Err(parse_error) => return Err(format!("the status document is not JSON: {parse_error}")),
    };
    let head_commit = git_head_commit(repository_root);
    for status_line in status_report_lines(&status_document, &head_commit) {
        println!("{status_line}");
    }
    println!(
        "  logs: {}",
        state_directory(channel_profile)?.join("logs").display()
    );
    Ok(())
}

/// The human status report. Pure, so its wording and the stale-build
/// warning are tested from a fixture status document.
pub(crate) fn status_report_lines(
    status_document: &serde_json::Value,
    head_commit: &str,
) -> Vec<String> {
    let mut status_lines = vec![
        format!(
            "Development instance: {} ({})",
            text_field(status_document, "/status"),
            text_field(status_document, "/activity")
        ),
        format!(
            "  model:             {}  [{}]",
            text_field(status_document, "/ready_model_id"),
            text_field(status_document, "/expert_memory_mode")
        ),
        format!(
            "  memory ceiling:    {} effective of {} (configured {}, pending {})",
            gigabyte_field(status_document, "/mlx_memory_ceiling_bytes"),
            gigabyte_field(status_document, "/machine_mlx_memory_ceiling_bytes"),
            text_field(status_document, "/configured_maximum_mlx_memory_gb"),
            text_field(status_document, "/pending_mlx_memory_ceiling_bytes")
        ),
        format!(
            "  session:           {} requests · {:.0} tok/s prefill · {:.0} tok/s decode",
            text_field(status_document, "/serving_session/completed_request_count"),
            number_field(
                status_document,
                "/serving_session/average_prefill_tok_per_second"
            ),
            number_field(
                status_document,
                "/serving_session/average_generation_tok_per_second"
            )
        ),
    ];
    status_lines.push(build_freshness_line(status_document, head_commit));
    status_lines
}

/// A daemon built from another commit serves old code; saying so here is
/// what stops an afternoon of testing the wrong build.
fn build_freshness_line(status_document: &serde_json::Value, head_commit: &str) -> String {
    let daemon_commit = text_field(status_document, "/application/commit");
    if daemon_commit == "-" {
        return "  build: unknown".to_owned();
    }
    if head_commit.starts_with(&daemon_commit) {
        return format!(
            "  build: {daemon_commit} (matches HEAD; dirty {})",
            text_field(status_document, "/application/is_dirty")
        );
    }
    format!(
        "  WARNING: built from {daemon_commit}, but HEAD is {head_commit}; rebuild with: cargo xtask app dev"
    )
}

fn text_field(status_document: &serde_json::Value, field_pointer: &str) -> String {
    match status_document.pointer(field_pointer) {
        None | Some(serde_json::Value::Null) => "-".to_owned(),
        Some(serde_json::Value::String(field_text)) => field_text.clone(),
        Some(field_value) => field_value.to_string(),
    }
}

fn number_field(status_document: &serde_json::Value, field_pointer: &str) -> f64 {
    match status_document.pointer(field_pointer) {
        Some(field_value) => field_value.as_f64().unwrap_or(0.0),
        None => 0.0,
    }
}

fn gigabyte_field(status_document: &serde_json::Value, field_pointer: &str) -> String {
    match status_document.pointer(field_pointer) {
        Some(field_value) if field_value.is_u64() => {
            format!(
                "{:.2} GB",
                field_value.as_f64().unwrap_or(0.0) / BYTES_PER_GIGABYTE
            )
        }
        _ => "-".to_owned(),
    }
}

fn git_head_commit(repository_root: &Path) -> String {
    match Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repository_root)
        .output()
    {
        Ok(git_output) if git_output.status.success() => {
            String::from_utf8_lossy(&git_output.stdout)
                .trim()
                .to_owned()
        }
        _ => "unknown".to_owned(),
    }
}

fn state_directory(channel_profile: &ChannelProfile) -> Result<PathBuf, String> {
    let Some(state_directory_name) = channel_profile.state_directory_name else {
        return Err("this channel keeps its state in Application Support".to_owned());
    };
    match std::env::var("HOME") {
        Ok(home_directory) => Ok(Path::new(&home_directory).join(state_directory_name)),
        Err(_) => Err("HOME is required to locate the instance state directory".to_owned()),
    }
}
