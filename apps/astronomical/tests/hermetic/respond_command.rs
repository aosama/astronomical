//! The `astronomical respond` one-shot journey: prompt in, streamed answer
//! out, process exits. The daemon side is a stub speaking the real framed
//! protocol over a real unix socket.

use std::{
    ffi::OsString,
    io::Write,
    path::PathBuf,
    process,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use astronomical_cli::errors::{RespondError, UsageError};
use astronomical_cli::{
    CliCommand, RespondArguments, RespondDependencies, parse_command, run_respond,
};
use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION,
    DaemonRequest, DaemonResponse, DaemonWorkerStatus, ProtocolReader, ProtocolWriter,
};
use tokio::time::timeout;

const RESPOND_TEST_TIMEOUT: Duration = Duration::from_secs(10);
const SOCKET_FILE_NAME: &str = "ipc.sock";

fn parse(arguments: &[&str]) -> Result<CliCommand, UsageError> {
    let process_arguments =
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from));
    parse_command(process_arguments)
}

fn fresh_respond_test_directory(test_name: &str) -> PathBuf {
    let nanos_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should provide time after the epoch")
        .as_nanos();
    let test_directory = std::env::temp_dir().join(format!(
        "ast-respond-{}-{}-{test_name}",
        process::id(),
        nanos_since_epoch % 1_000_000_000
    ));
    std::fs::create_dir_all(&test_directory)
        .expect("the respond test directory should be creatable");
    test_directory
}

fn respond_arguments(prompt: &str, model_id: Option<&str>, no_stream: bool) -> RespondArguments {
    RespondArguments {
        prompt: prompt.to_owned(),
        model_id: model_id.map(str::to_owned),
        no_stream,
    }
}

fn respond_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> RespondDependencies<'a> {
    RespondDependencies {
        candidate_socket_paths,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: RESPOND_TEST_TIMEOUT,
    }
}

/// Stub daemon answering the real framed protocol: handshake, status, and one
/// scripted chat generation whose terminal frame closes the connection.
fn spawn_stub_respond_daemon(
    socket_path: PathBuf,
    ready_model_id: Option<String>,
    answer_fragments: Vec<String>,
) -> tokio::task::JoinHandle<()> {
    let ready_model_id = std::sync::Arc::new(ready_model_id);
    let answer_fragments = std::sync::Arc::new(answer_fragments);
    // Bind before spawning so the client can never race an unbound socket.
    let std_listener = std::os::unix::net::UnixListener::bind(&socket_path)
        .expect("the stub daemon should bind the test socket");
    std_listener
        .set_nonblocking(true)
        .expect("the stub listener should accept non-blocking mode");
    let unix_listener = tokio::net::UnixListener::from_std(std_listener)
        .expect("the stub listener should register with the runtime");
    tokio::spawn(async move {
        loop {
            let Ok((connection_stream, _peer_address)) = unix_listener.accept().await else {
                return;
            };
            let ready_model_id = std::sync::Arc::clone(&ready_model_id);
            let answer_fragments = std::sync::Arc::clone(&answer_fragments);
            tokio::spawn(async move {
                let (read_half, write_half) = connection_stream.into_split();
                let mut protocol_reader = ProtocolReader::new(read_half);
                let mut protocol_writer = ProtocolWriter::new(write_half);
                loop {
                    let Some(daemon_request) = protocol_reader.next_daemon_request().await.expect(
                        "the stub daemon request read should not fail at the transport layer",
                    ) else {
                        return;
                    };
                    match daemon_request {
                        DaemonRequest::Handshake => {
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::HandshakeAccepted {
                                    protocol_version: DAEMON_PROTOCOL_VERSION,
                                    application_name: DAEMON_APPLICATION_NAME.to_owned(),
                                })
                                .await
                                .expect("the stub handshake should transmit");
                        }
                        DaemonRequest::Status => {
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::Status {
                                    worker_status: DaemonWorkerStatus::Ready,
                                    ready_model_id: ready_model_id.as_ref().clone(),
                                })
                                .await
                                .expect("the stub status should transmit");
                        }
                        DaemonRequest::ChatGenerate { .. } => {
                            for answer_fragment in answer_fragments.iter() {
                                protocol_writer
                                    .send_daemon_response(&DaemonResponse::ChatGenerationText {
                                        text: answer_fragment.clone(),
                                    })
                                    .await
                                    .expect("the stub fragment should transmit");
                            }
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::ChatGenerationCompleted {
                                    prompt_token_count: 7,
                                    generated_token_count: 2,
                                    reasoning_token_count: 0,
                                    cached_token_count: 0,
                                    reason: ChatGenerationCompletionReason::EndOfSequence,
                                })
                                .await
                                .expect("the stub completion frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                        DaemonRequest::EmbedGenerate { .. } => {
                            panic!("the respond stub daemon must not receive embeddings requests");
                        }
                    }
                }
            });
        }
    })
}

#[test]
fn should_parse_respond_with_a_bare_prompt() {
    let parsed_command =
        parse(&["respond", "Hello there"]).expect("a bare respond prompt should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments("Hello there", None, false))
    );
}

#[test]
fn should_parse_respond_prompt_with_model_and_no_stream() {
    let parsed_command = parse(&[
        "respond",
        "Hello there",
        "--model",
        "test/model",
        "--no-stream",
    ])
    .expect("respond with --model and --no-stream should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Respond(respond_arguments("Hello there", Some("test/model"), true))
    );
}

#[test]
fn should_reject_respond_without_a_prompt_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond"]),
        Err(UsageError::RespondPromptRequired)
    ));
}

#[test]
fn should_reject_respond_with_an_unknown_argument_as_a_usage_error() {
    assert!(matches!(
        parse(&["respond", "Hi", "--bogus"]),
        Err(UsageError::UnknownArgument(argument)) if argument == "--bogus"
    ));
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running() {
    let test_directory = fresh_respond_test_directory("not-running");
    let missing_socket_path = test_directory.join(SOCKET_FILE_NAME);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![missing_socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        RESPOND_TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    assert!(
        matches!(respond_outcome, Err(RespondError::DaemonNotRunning)),
        "a missing socket must surface as the not-running error: {respond_outcome:?}"
    );
    assert!(
        format!("{respond_outcome:?}").contains("Astronomical isn't running"),
        "the not-running error should tell the user to start Astronomical: {respond_outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_stream_generated_text_to_stdout() {
    let test_directory = fresh_respond_test_directory("stream-text");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_respond_daemon(
        socket_path.clone(),
        Some("test/local-chatter".to_owned()),
        vec!["Hello".to_owned(), " world".to_owned()],
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        RESPOND_TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    respond_outcome.expect("the respond journey should complete against the stub daemon");
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "Hello world",
        "the streamed fragments should appear on stdout in order"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_print_the_finished_answer_once_with_no_stream() {
    let test_directory = fresh_respond_test_directory("no-stream");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_respond_daemon(
        socket_path.clone(),
        Some("test/local-chatter".to_owned()),
        vec!["Hello".to_owned(), " world".to_owned()],
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        RESPOND_TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", None, true),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    respond_outcome.expect("the respond journey should complete against the stub daemon");
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "Hello world",
        "--no-stream should print the finished answer exactly once"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_when_the_requested_model_is_not_loaded() {
    let test_directory = fresh_respond_test_directory("wrong-model");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_respond_daemon(
        socket_path.clone(),
        Some("test/other-model".to_owned()),
        vec![],
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        RESPOND_TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", Some("test/wanted-model"), false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    assert!(
        matches!(
            &respond_outcome,
            Err(RespondError::RequestedModelNotReady {
                requested_model_id,
                ready_model_id,
            }) if requested_model_id == "test/wanted-model"
                && ready_model_id == "test/other-model"
        ),
        "requesting a model that is not loaded must fail with both identities: {respond_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_when_no_model_is_loaded() {
    let test_directory = fresh_respond_test_directory("no-model");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_respond_daemon(socket_path.clone(), None, vec![]);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        RESPOND_TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    assert!(
        matches!(respond_outcome, Err(RespondError::NoModelLoaded)),
        "a ready daemon with no resident model must fail with the no-model error: {respond_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
