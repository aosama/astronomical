//! `astronomical launch claude` end to end (issue #891): a real daemon on a
//! real model, the launch verb's own preparation, and the real Claude Code
//! harness completing a chat turn against the local Messages endpoint.
//!
//! The launch verb is `prepare_launch` followed by an exec of the program,
//! arguments, and environment it prepares. The journey runs exactly that,
//! with the daemon's test address injected as the instance candidate,
//! because the installed CLI binary only targets its channel's fixed port.
//!
//! The assertions cover what the launch verb owns: the turn completes, the
//! local model serves it, and Claude Code uses the context window the daemon
//! advertises instead of its 200K guess for an unrecognized model. Whether a
//! model chooses to call a tool is a property of the model, not the launch.

use std::{
    ffi::OsString,
    io::Cursor,
    time::{Duration, Instant},
};

use astronomical_cli::{LaunchArguments, LaunchDependencies, PreparedLaunch, prepare_launch};
use astronomical_config::{DiscoveredModel, ModelCapabilities};
use serial_test::serial;
use tokio::time::timeout;

use super::openai_rest::{launch_serving_rest_server_for_model, stop_serving_rest_server};
use crate::small_dense_model;

/// SSD-paging e2e journeys get a 120-second runner budget; this journey
/// stays inside it with room for teardown.
const CLAUDE_CODE_TURN_TIMEOUT: Duration = Duration::from_secs(110);
const SONNET_PROMPT: &str = "Complete this line by Shakespeare in one short sentence: Shall I compare thee to a summer's day?";

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "launches a real daemon and model and needs Claude Code installed on PATH"]
async fn should_complete_a_claude_code_turn_on_the_local_model_through_the_launch_verb() {
    let journey_started_at = Instant::now();
    let launch_model = small_dense_model::configured_deployment_litmus_model();
    let serving_rest_server = launch_serving_rest_server_for_model(
        &launch_model.model_id,
        launch_model.model_directory.clone(),
        None,
        None,
    )
    .await;
    let daemon_ready_elapsed = journey_started_at.elapsed();
    let launch_started_at = Instant::now();
    let prepared_launch =
        prepare_claude_code_launch(serving_rest_server.server_address, &launch_model.model_id);
    let launch_preparation_elapsed = launch_started_at.elapsed();
    let turn_started_at = Instant::now();
    let turn_result = run_prepared_claude_code(&prepared_launch).await;
    let harness_turn_elapsed = turn_started_at.elapsed();
    stop_serving_rest_server(serving_rest_server).await;
    report_launch_journey_timings(&LaunchJourneyTimings {
        daemon_ready_elapsed,
        launch_preparation_elapsed,
        harness_turn_elapsed,
        total_elapsed: journey_started_at.elapsed(),
    });
    assert_turn_served_by_the_local_model(&turn_result, &launch_model);
}

/// The turn finished without error, the local model produced it, and Claude
/// Code sized its context to the window the daemon advertised.
fn assert_turn_served_by_the_local_model(
    turn_result: &serde_json::Value,
    launch_model: &DiscoveredModel,
) {
    assert_eq!(
        turn_result["is_error"], false,
        "Claude Code reported an error: {turn_result}"
    );
    assert!(
        !turn_result["result"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "the turn should produce an answer: {turn_result}"
    );
    let local_model_usage = &turn_result["modelUsage"][&launch_model.model_id];
    assert!(
        local_model_usage["inputTokens"].as_u64().unwrap_or(0) > 0
            && local_model_usage["outputTokens"].as_u64().unwrap_or(0) > 0,
        "the local model should serve the turn: {turn_result}"
    );
    assert_eq!(
        local_model_usage["contextWindow"].as_u64(),
        advertised_context_window(launch_model),
        "Claude Code should use the window the launch declared: {turn_result}"
    );
}

fn advertised_context_window(launch_model: &DiscoveredModel) -> Option<u64> {
    match &launch_model.capabilities {
        ModelCapabilities::Chat(chat_capabilities) => {
            Some(u64::from(chat_capabilities.context_window))
        }
        _ => None,
    }
}

/// The launch verb's preparation, with a headless one-turn Claude Code
/// invocation passed through after `--`.
fn prepare_claude_code_launch(
    daemon_address: std::net::SocketAddr,
    model_id: &str,
) -> PreparedLaunch {
    let mut stdin = Cursor::new(Vec::new());
    let mut stderr_bytes = Vec::new();
    let mut harness_arguments = Vec::new();
    for harness_argument in [
        "-p",
        SONNET_PROMPT,
        "--max-turns",
        "2",
        "--output-format",
        "json",
    ] {
        harness_arguments.push(OsString::from(harness_argument));
    }
    prepare_launch(
        LaunchArguments {
            tool_slug: Some("claude".to_owned()),
            model_id: Some(model_id.to_owned()),
            harness_arguments,
        },
        LaunchDependencies {
            candidate_bind_addresses: vec![daemon_address],
            path_value: std::env::var_os("PATH").unwrap_or_default(),
            is_interactive: false,
            stdin: &mut stdin,
            stderr: &mut stderr_bytes,
            http_timeout: Duration::from_secs(5),
        },
    )
    .expect("the launch verb should prepare Claude Code against the running daemon")
}

/// Runs the prepared program exactly as the launch verb's exec does: its
/// arguments and its process-scoped environment, nothing else.
async fn run_prepared_claude_code(prepared_launch: &PreparedLaunch) -> serde_json::Value {
    let working_directory = tempfile::tempdir().expect("a working directory for Claude Code");
    let mut claude_code_command = tokio::process::Command::new(&prepared_launch.program);
    claude_code_command
        .args(&prepared_launch.arguments)
        .current_dir(working_directory.path())
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    for (environment_name, environment_value) in &prepared_launch.extra_environment {
        claude_code_command.env(environment_name, environment_value);
    }
    let command_output = timeout(CLAUDE_CODE_TURN_TIMEOUT, claude_code_command.output())
        .await
        .expect("the Claude Code turn should finish within its budget")
        .expect("Claude Code should start");
    let printed_output = String::from_utf8_lossy(&command_output.stdout);
    match serde_json::from_str(printed_output.trim()) {
        Ok(turn_result) => turn_result,
        Err(json_error) => {
            panic!("Claude Code should print one JSON result ({json_error}): {printed_output}")
        }
    }
}

/// Per-stage wall-clock attribution for the journey, so a slow run shows
/// whether the daemon, the launch preparation, or the harness turn grew.
struct LaunchJourneyTimings {
    daemon_ready_elapsed: Duration,
    launch_preparation_elapsed: Duration,
    harness_turn_elapsed: Duration,
    total_elapsed: Duration,
}

fn report_launch_journey_timings(launch_journey_timings: &LaunchJourneyTimings) {
    eprintln!(
        "[timing] claude_code_launch daemon_ready_ms={} launch_preparation_ms={} harness_turn_ms={} total_ms={}",
        launch_journey_timings.daemon_ready_elapsed.as_millis(),
        launch_journey_timings
            .launch_preparation_elapsed
            .as_millis(),
        launch_journey_timings.harness_turn_elapsed.as_millis(),
        launch_journey_timings.total_elapsed.as_millis()
    );
}
