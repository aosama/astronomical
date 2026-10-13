//! Whole-product acceptance journey: one `respond --schema`-shaped chat
//! generation through the daemon's IPC socket.
//!
//! Driven in two fidelities selected by `ASTRONOMICAL_REAL_MODEL=1`: the
//! real `astronomicald` daemon (which spawns the real inference worker from
//! its own directory), configured to discover the installed Qwen3.5-2B-4bit
//! artifact — or, by default, the fixture `astronomical-daemon-test-daemon`
//! with the scripted worker, which needs no artifacts so the suite runs on
//! any machine. Both fidelities send one `DaemonRequest::ChatGenerate`
//! carrying raw `--schema` JSON — the exact request the CLI's `respond
//! --schema` verb sends — through the daemon's schema validation into
//! logits-masked structured generation, and both assert the same reply
//! contract: a single JSON object matching the supplied schema.
//!
//! Real-fidelity prerequisites: the real worker binary must sit next to the
//! daemon binary, so build both before running (`cargo build
//! -p astronomical-daemon -p astronomical-inference-worker`), and the
//! configured Development model directories must contain the installed
//! Qwen3.5-2B-4bit artifact.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DaemonIpcClient, DaemonRequest, DaemonResponse,
};
use astronomical_rest_contract::extract_json_value_from_text;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

use crate::common::fidelity::ModelFidelity;

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const DAEMON_STARTUP_WAIT: Duration = Duration::from_secs(20);
/// How long the socket poll and the worker boot-handshake retries each wait.
/// Matches the image journey's proven endpoint-wait budget; a fully loaded
/// continuous-integration runner boots the worker more slowly than an idle one.
const WORKER_BOOT_WAIT: Duration = Duration::from_secs(30);
const FRAME_WAIT: Duration = Duration::from_secs(90);
const DAEMON_DIAGNOSTICS_TAIL_LINES: usize = 20;
const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";
const CHAT_MODEL_ID: &str = "Qwen3.5-2B-4bit";
/// Thinking-enabled Qwen otherwise fills max_tokens with reasoning and returns
/// an empty JSON answer; the same budget the REST structured-output journey uses.
const THINKING_TOKEN_BUDGET: u32 = 64;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 384;
/// Keys are in serde_json serialization order, so the daemon's canonical
/// constraint text equals this text byte for byte.
const HOUSEHOLD_SCHEMA_JSON: &str = "{\"properties\":{\"household\":{\"type\":\"string\"},\"speaker\":{\"type\":\"string\"}},\"required\":[\"household\",\"speaker\"],\"type\":\"object\"}";
/// A prompt grounded in the repository's mandated Romeo and Juliet source.
const JOURNEY_PROMPT: &str = "Who speaks the line \"O Romeo, Romeo, wherefore art thou Romeo?\", and which household does that speaker belong to? Reply with the JSON object.";

#[tokio::test]
async fn should_serve_one_schema_matching_json_object_over_the_cli_ipc_path() {
    timeout(JOURNEY_TIMEOUT, async {
        let fidelity = ModelFidelity::from_environment();
        let state_directory = tempfile::tempdir().expect("state directory should be created");
        let isolated_home = tempfile::tempdir().expect("isolated home should be created");
        write_instance_config(state_directory.path(), &journey_scan_root(fidelity));

        let (mut daemon_process, daemon_diagnostics_tail) =
            spawn_daemon(state_directory.path(), isolated_home.path(), fidelity).await;
        let socket_path = state_directory.path().join("ipc.sock");
        wait_for_ipc_socket(&socket_path).await;

        let reply_text = match send_chat_generate_with_schema_with_startup_retries(
            &socket_path,
            HOUSEHOLD_SCHEMA_JSON,
        )
        .await
        {
            Ok(reply_text) => reply_text,
            Err(generation_failure) => {
                    let _unused_termination_result = daemon_process.start_kill();
                    let diagnostics_tail = daemon_diagnostics_tail
                        .await
                        .unwrap_or_default();
                    panic!(
                        "the schema journey failed: {generation_failure}; daemon diagnostics tail: {diagnostics_tail:?}"
                    );
                }
            };

        terminate_daemon(&daemon_process);
        let daemon_exit = daemon_process
            .wait()
            .await
            .expect("the daemon should be reaped after termination");
        assert!(
            daemon_exit.success(),
            "the daemon should shut down cleanly, exited with {daemon_exit}"
        );

        let parsed_reply = extract_json_value_from_text(&reply_text)
            .unwrap_or_else(|| panic!("the reply must be one JSON object, received {reply_text:?}"));
        let reply_object = parsed_reply
            .as_object()
            .unwrap_or_else(|| panic!("the reply must be a single JSON object, received {reply_text:?}"));
        assert_eq!(
            reply_object.len(),
            2,
            "the schema has exactly two required properties, received {reply_object:?}"
        );
        for required_property in ["speaker", "household"] {
            assert!(
                reply_object
                    .get(required_property)
                    .is_some_and(|property_value| property_value.as_str().is_some_and(
                        |property_text| !property_text.trim().is_empty()
                    )),
                "the reply must carry a non-empty string for {required_property:?}, received {reply_object:?}"
            );
        }
    })
    .await
    .expect("the live IPC schema journey should finish within its 115 s budget");
}

/// Resolves the installed chat model the same way the worker serving journeys
/// do, from the Development configuration, so no developer path is hardcoded.
fn installed_chat_model_directory() -> PathBuf {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the Development Astronomical configuration should load for model lookup");
    astronomical_config
        .find_configured_model_directory_by_id(CHAT_MODEL_ID)
        .expect("the configured model directories should be searchable")
        .unwrap_or_else(|| {
            panic!(
                "the Development Astronomical configuration model_directories should discover \
                 model ID {CHAT_MODEL_ID}"
            )
        })
}

/// The instance config's model scan root per fidelity: the real daemon
/// discovers the installed artifact through it; the fixture daemon ignores
/// config.json, so any well-formed root keeps the document byte-identical.
fn journey_scan_root(fidelity: ModelFidelity) -> String {
    match fidelity {
        ModelFidelity::Real => scan_root(&installed_chat_model_directory()),
        ModelFidelity::Scripted => "fixtures/models/scripted-chat".to_owned(),
    }
}

/// Discovery accepts the parent of the resolved model directory and finds the
/// artifact under it.
fn scan_root(model_directory: &Path) -> String {
    model_directory
        .parent()
        .expect("the model directory should have a parent scan root")
        .to_string_lossy()
        .into_owned()
}

/// The daemon resolves configuration from the state directory it is given.
fn write_instance_config(state_directory: &Path, model_root: &str) {
    let configuration_document = format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":["{model_root}"]}},"chunking":{{"fixed_prompt_processing_chunk_size_tokens":2048}}}}"#
    );
    std::fs::write(state_directory.join("config.json"), configuration_document)
        .expect("instance config should be written");
}

/// Spawns the daemon for the journey's fidelity: the real `astronomicald`
/// (which discovers and spawns its own real worker) or the fixture
/// `astronomical-daemon-test-daemon` pointed at the scripted worker. The
/// fixture pair answers the same IPC surface with the same startup line.
async fn spawn_daemon(
    state_directory: &Path,
    isolated_home: &Path,
    fidelity: ModelFidelity,
) -> (Child, tokio::task::JoinHandle<Vec<String>>) {
    let daemon_executable_path = match fidelity {
        ModelFidelity::Real => {
            let daemon_executable_path =
                astronomical_test_harness::workspace_binary("astronomicald")
                    .to_string_lossy()
                    .into_owned();
            let daemon_directory = Path::new(&daemon_executable_path)
                .parent()
                .expect("the daemon executable should have a parent directory");
            assert!(
                daemon_directory
                    .join("astronomical-inference-worker")
                    .is_file(),
                "the real worker binary must exist next to the daemon: build it first with \
                 `cargo build -p astronomical-inference-worker`"
            );
            daemon_executable_path
        }
        ModelFidelity::Scripted => {
            astronomical_test_harness::workspace_binary("astronomical-daemon-test-daemon")
                .to_string_lossy()
                .into_owned()
        }
    };
    let mut daemon_spawn_command = Command::new(daemon_executable_path);
    daemon_spawn_command
        .args(["--instance", "development", "--state-directory"])
        .arg(state_directory)
        .env("HOME", isolated_home)
        .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
        .env_remove("ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if fidelity == ModelFidelity::Scripted {
        daemon_spawn_command
            .env(
                "ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH",
                astronomical_test_harness::workspace_binary("astronomical-daemon-test-worker"),
            )
            .env("ASTRONOMICAL_TEST_WORKER_READY_MODEL_ID", CHAT_MODEL_ID);
    }
    let mut daemon_process = daemon_spawn_command
        .spawn()
        .expect("the daemon process should start");
    // The daemon spawns its worker at boot; a failed boot spawn poisons every
    // later generation request with a fast rejection, so the diagnostics tail
    // is the only way to see why. Keep a bounded tail for the failure path.
    let daemon_stderr = daemon_process
        .stderr
        .take()
        .expect("daemon stderr should be captured");
    let daemon_diagnostics = tokio::spawn(async move {
        let mut diagnostics_tail = Vec::new();
        let mut stderr_lines = BufReader::new(daemon_stderr).lines();
        while let Ok(Some(diagnostic_line)) = stderr_lines.next_line().await {
            if diagnostics_tail.len() == DAEMON_DIAGNOSTICS_TAIL_LINES {
                diagnostics_tail.remove(0);
            }
            diagnostics_tail.push(diagnostic_line);
        }
        diagnostics_tail
    });
    let daemon_stdout = daemon_process
        .stdout
        .take()
        .expect("daemon stdout should be captured");
    let mut stdout_lines = BufReader::new(daemon_stdout).lines();
    let startup_line = timeout(DAEMON_STARTUP_WAIT, stdout_lines.next_line())
        .await
        .expect("the daemon should report its endpoint")
        .expect("daemon startup output should be readable")
        .expect("daemon startup output should not end");
    assert!(
        startup_line.starts_with(DAEMON_STARTUP_PREFIX),
        "the daemon should report the expected startup prefix, received {startup_line:?}"
    );
    (daemon_process, daemon_diagnostics)
}

/// The IPC socket file appears when the daemon binds its listener; poll for it
/// rather than sleeping a fixed interval.
async fn wait_for_ipc_socket(socket_path: &Path) {
    let socket_wait_deadline = tokio::time::Instant::now() + WORKER_BOOT_WAIT;
    while !socket_path.exists() {
        assert!(
            tokio::time::Instant::now() < socket_wait_deadline,
            "the daemon should bind its IPC socket at {socket_path:?} inside the wait budget"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The daemon boots its worker asynchronously while already serving, so the
/// first schema-carrying request can race the boot handshake; retry until the
/// worker is live, exactly like the image journey's startup retries. The error
/// carries the last rejection so the caller can attach daemon diagnostics.
async fn send_chat_generate_with_schema_with_startup_retries(
    socket_path: &Path,
    schema_json: &str,
) -> Result<String, String> {
    let startup_deadline = tokio::time::Instant::now() + WORKER_BOOT_WAIT;
    loop {
        let generation_failure =
            match send_chat_generate_with_schema(socket_path, schema_json).await {
                Ok(reply_text) => return Ok(reply_text),
                Err(generation_failure) => generation_failure,
            };
        if !generation_failure.contains("worker is unavailable") {
            return Err(generation_failure);
        }
        if tokio::time::Instant::now() >= startup_deadline {
            return Err(format!(
                "the worker did not complete its boot handshake inside {WORKER_BOOT_WAIT:?}; \
                 latest failure: {generation_failure}"
            ));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Performs the handshake on one connection, then sends the schema-carrying
/// chat request on a fresh connection per the wire contract, and accumulates
/// every visible text fragment until the terminal frame.
async fn send_chat_generate_with_schema(
    socket_path: &Path,
    schema_json: &str,
) -> Result<String, String> {
    let mut handshake_client = DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the handshake client should connect to the daemon IPC service");
    timeout(
        FRAME_WAIT,
        handshake_client.send_request(&DaemonRequest::Handshake),
    )
    .await
    .expect("the handshake send should finish inside the frame budget")
    .expect("the handshake request should transmit");
    let _handshake_accepted = timeout(FRAME_WAIT, handshake_client.next_response())
        .await
        .expect("the handshake response should arrive inside the frame budget")
        .expect("the handshake response read should not fail at the transport layer")
        .expect("the daemon should answer the handshake");
    drop(handshake_client);

    let mut request_client = DaemonIpcClient::connect(socket_path.to_path_buf())
        .await
        .expect("the request client should connect to the daemon IPC service");
    timeout(
        FRAME_WAIT,
        request_client.send_request(&DaemonRequest::ChatGenerate {
            model: CHAT_MODEL_ID.to_owned(),
            messages: vec![ChatMessage::User {
                content: JOURNEY_PROMPT.to_owned(),
                images: Vec::new(),
            }],
            settings: ChatGenerationSettings {
                max_output_tokens: MAXIMUM_OUTPUT_TOKEN_COUNT as u16,
                temperature_thousandths: None,
                top_p_thousandths: None,
                seed: None,
                thinking_budget: Some(THINKING_TOKEN_BUDGET as u16),
            },
            schema_json: Some(schema_json.to_owned()),
        }),
    )
    .await
    .expect("the chat request should send inside the frame budget")
    .expect("the chat request should transmit");

    let mut reply_text = String::new();
    loop {
        let daemon_frame = timeout(FRAME_WAIT, request_client.next_response())
            .await
            .expect("the response frame should arrive inside the frame budget")
            .expect("the response frame read should not fail at the transport layer");
        match daemon_frame {
            Some(DaemonResponse::ChatGenerationText { text }) => reply_text.push_str(&text),
            Some(DaemonResponse::ChatGenerationCompleted {
                generated_token_count,
                ..
            }) => {
                assert!(
                    generated_token_count > 0,
                    "a schema-constrained reply must generate tokens"
                );
                return Ok(reply_text);
            }
            Some(DaemonResponse::ChatGenerationFailed { reason }) => {
                return Err(format!(
                    "the schema-constrained generation failed: {reason:?}"
                ));
            }
            Some(DaemonResponse::GenerationRejected { reason }) => {
                return Err(format!(
                    "the daemon rejected the schema-carrying request: {reason}"
                ));
            }
            Some(DaemonResponse::RequestRejected { reason }) => {
                return Err(format!("the daemon declined the chat request: {reason}"));
            }
            Some(_) => {}
            None => {
                return Err(
                    "the daemon closed the IPC connection before completing the reply".to_owned(),
                );
            }
        }
    }
}

fn terminate_daemon(daemon_process: &Child) {
    let process_id = daemon_process.id().expect("the daemon should still run");
    let terminate_status = std::process::Command::new("kill")
        .args(["-TERM", &process_id.to_string()])
        .status()
        .expect("the test should send SIGTERM");
    assert!(terminate_status.success());
}
