//! The `astronomical embed` one-shot journey: text, file, or stdin in, one
//! JSON vector document on stdout, process exits. The daemon side is a stub
//! speaking the real framed protocol over a real unix socket.

use std::{
    ffi::OsString,
    io::{Read, Write},
    path::PathBuf,
    process,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use astronomical_cli::errors::{EmbedError, UsageError};
use astronomical_cli::{CliCommand, EmbedArguments, EmbedDependencies, parse_command, run_embed};
use astronomical_ipc_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonRequest, DaemonResponse,
    DaemonWorkerStatus, EmbeddingsFailureReason, ProtocolReader, ProtocolWriter,
};
use tokio::time::timeout;

const EMBED_TEST_TIMEOUT: Duration = Duration::from_secs(10);
const SOCKET_FILE_NAME: &str = "ipc.sock";

fn parse(arguments: &[&str]) -> Result<CliCommand, UsageError> {
    let process_arguments =
        std::iter::once(OsString::from("astronomical")).chain(arguments.iter().map(OsString::from));
    parse_command(process_arguments)
}

fn embed_arguments(
    text: Option<&str>,
    file_path: Option<PathBuf>,
    model_id: Option<&str>,
) -> EmbedArguments {
    EmbedArguments {
        text: text.map(str::to_owned),
        file_path,
        model_id: model_id.map(str::to_owned),
    }
}

fn embed_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdin: &'a mut dyn Read,
    stdout: &'a mut Vec<u8>,
) -> EmbedDependencies<'a> {
    EmbedDependencies {
        candidate_socket_paths,
        stdin,
        stdout: stdout as &mut dyn Write,
        request_timeout: EMBED_TEST_TIMEOUT,
    }
}

fn fresh_embed_test_directory(test_name: &str) -> PathBuf {
    let nanos_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should provide time after the epoch")
        .as_nanos();
    let test_directory = std::env::temp_dir().join(format!(
        "ast-embed-{}-{}-{test_name}",
        process::id(),
        nanos_since_epoch % 1_000_000_000
    ));
    std::fs::create_dir_all(&test_directory).expect("the embed test directory should be creatable");
    test_directory
}

/// Outcome the stub daemon serves for one embeddings request.
#[derive(Clone, Copy)]
enum StubEmbeddingsOutcome {
    Completed,
    ContextLengthExceeded,
}

/// Stub daemon answering the real framed protocol: handshake, status, and
/// one embeddings result frame per embeddings request.
fn spawn_stub_embed_daemon(
    socket_path: PathBuf,
    ready_model_id: Option<String>,
    embeddings_outcome: StubEmbeddingsOutcome,
) -> tokio::task::JoinHandle<()> {
    let ready_model_id = std::sync::Arc::new(ready_model_id);
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
                        DaemonRequest::EmbedGenerate { .. } => {
                            let embeddings_response = match embeddings_outcome {
                                StubEmbeddingsOutcome::Completed => {
                                    DaemonResponse::EmbeddingsCompleted {
                                        model: ready_model_id
                                            .as_ref()
                                            .clone()
                                            .expect("the completed stub always has a model"),
                                        vectors: vec![vec![0.25, -0.5]],
                                        input_token_counts: vec![3],
                                    }
                                }
                                StubEmbeddingsOutcome::ContextLengthExceeded => {
                                    DaemonResponse::EmbeddingsFailed {
                                        reason: EmbeddingsFailureReason::ContextLengthExceeded {
                                            actual_total_context_tokens: 5_000,
                                            maximum_context_tokens: 2_048,
                                        },
                                    }
                                }
                            };
                            protocol_writer
                                .send_daemon_response(&embeddings_response)
                                .await
                                .expect("the stub embeddings frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                        DaemonRequest::ChatGenerate { .. } => {
                            panic!("the embed journey must never send a chat generation request");
                        }
                    }
                }
            });
        }
    })
}

#[test]
fn should_parse_embed_with_a_bare_text_argument() {
    let parsed_command = parse(&["embed", "hello world"]).expect("a bare embed text should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(Some("hello world"), None, None))
    );
}

#[test]
fn should_parse_embed_with_model_and_file() {
    let parsed_command = parse(&[
        "embed",
        "--model",
        "test/local-embedder",
        "--file",
        "input.txt",
    ])
    .expect("embed with --model and --file should parse");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(
            None,
            Some(PathBuf::from("input.txt")),
            Some("test/local-embedder")
        ))
    );
}

#[test]
fn should_parse_embed_with_no_input_for_stdin_mode() {
    let parsed_command = parse(&["embed"]).expect("embed without input should parse for stdin");
    assert_eq!(
        parsed_command,
        CliCommand::Embed(embed_arguments(None, None, None))
    );
}

#[test]
fn should_reject_embed_with_an_unknown_argument_as_a_usage_error() {
    assert!(matches!(
        parse(&["embed", "hello", "--unknown"]),
        Err(UsageError::UnknownArgument(unknown_argument)) if unknown_argument == "--unknown"
    ));
}

#[test]
fn should_reject_embed_with_both_text_and_file_as_a_usage_error() {
    assert!(matches!(
        parse(&["embed", "hello", "--file", "input.txt"]),
        Err(UsageError::EmbedInputConflict)
    ));
}

#[tokio::test]
async fn should_embed_a_text_argument_into_one_json_vector_document() {
    let test_directory = fresh_embed_test_directory("text-argument");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_embed_daemon(
        socket_path.clone(),
        Some("test/local-embedder".to_owned()),
        StubEmbeddingsOutcome::Completed,
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut embed_dependencies =
        embed_dependencies(vec![socket_path], &mut stdin_content, &mut stdout);

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(
            &embed_arguments(Some("hello"), None, None),
            &mut embed_dependencies,
        ),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    embed_outcome.expect("the embed journey should complete against the stub daemon");

    let vector_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the embed payload should be exactly one JSON document");
    assert_eq!(
        vector_document["model"], "test/local-embedder",
        "the vector document should name the model that produced it"
    );
    assert_eq!(
        vector_document["embedding"],
        serde_json::json!([0.25, -0.5]),
        "the vector document should carry the embedding vector"
    );
    assert_eq!(
        vector_document["input_tokens"], 3,
        "the vector document should report the input token count"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_embed_text_from_a_file() {
    let test_directory = fresh_embed_test_directory("file-input");
    let input_file_path = test_directory.join("input.txt");
    std::fs::write(&input_file_path, "hello from a file")
        .expect("the embed input file should be writable");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_embed_daemon(
        socket_path.clone(),
        Some("test/local-embedder".to_owned()),
        StubEmbeddingsOutcome::Completed,
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut embed_dependencies =
        embed_dependencies(vec![socket_path], &mut stdin_content, &mut stdout);

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(
            &embed_arguments(None, Some(input_file_path), None),
            &mut embed_dependencies,
        ),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    embed_outcome.expect("the embed journey should complete against the stub daemon");

    let vector_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the file-input embed payload should be exactly one JSON document");
    assert_eq!(vector_document["model"], "test/local-embedder");
    assert_eq!(
        vector_document["embedding"],
        serde_json::json!([0.25, -0.5])
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_embed_text_from_stdin() {
    let test_directory = fresh_embed_test_directory("stdin-input");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_embed_daemon(
        socket_path.clone(),
        Some("test/local-embedder".to_owned()),
        StubEmbeddingsOutcome::Completed,
    );
    let mut stdin_content = std::io::Cursor::new(b"hello from stdin".to_vec());
    let mut stdout = Vec::new();
    let mut embed_dependencies =
        embed_dependencies(vec![socket_path], &mut stdin_content, &mut stdout);

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(&embed_arguments(None, None, None), &mut embed_dependencies),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    embed_outcome.expect("the embed journey should complete against the stub daemon");

    let vector_document: serde_json::Value = serde_json::from_slice(&stdout)
        .expect("the stdin embed payload should be exactly one JSON document");
    assert_eq!(vector_document["model"], "test/local-embedder");
    assert_eq!(
        vector_document["embedding"],
        serde_json::json!([0.25, -0.5])
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running_for_embed() {
    let test_directory = fresh_embed_test_directory("not-running");
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![test_directory.join(SOCKET_FILE_NAME)],
        &mut stdin_content,
        &mut stdout,
    );

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(
            &embed_arguments(Some("hello"), None, None),
            &mut embed_dependencies,
        ),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    assert!(
        matches!(embed_outcome, Err(EmbedError::DaemonNotRunning)),
        "a missing daemon must fail with the not-running error: {embed_outcome:?}"
    );
    assert!(
        format!("{embed_outcome:?}").contains("Astronomical isn't running"),
        "the not-running error must carry the clear guidance text: {embed_outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_reject_embed_when_no_model_is_resident_and_none_was_requested() {
    let test_directory = fresh_embed_test_directory("no-model");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task =
        spawn_stub_embed_daemon(socket_path.clone(), None, StubEmbeddingsOutcome::Completed);
    let mut stdin_content = std::io::Cursor::new(b"hello".to_vec());
    let mut stdout = Vec::new();
    let mut embed_dependencies =
        embed_dependencies(vec![socket_path], &mut stdin_content, &mut stdout);

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(&embed_arguments(None, None, None), &mut embed_dependencies),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    assert!(
        matches!(embed_outcome, Err(EmbedError::NoModelLoaded)),
        "no resident model and no requested model must fail with the no-model error: {embed_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_with_the_worker_reason_when_embeddings_fail() {
    let test_directory = fresh_embed_test_directory("worker-failure");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_embed_daemon(
        socket_path.clone(),
        Some("test/local-embedder".to_owned()),
        StubEmbeddingsOutcome::ContextLengthExceeded,
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut embed_dependencies =
        embed_dependencies(vec![socket_path], &mut stdin_content, &mut stdout);

    let embed_outcome = timeout(
        EMBED_TEST_TIMEOUT,
        run_embed(
            &embed_arguments(Some("a very long text"), None, None),
            &mut embed_dependencies,
        ),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    assert!(
        matches!(
            &embed_outcome,
            Err(EmbedError::EmbeddingsFailed {
                reason: EmbeddingsFailureReason::ContextLengthExceeded {
                    actual_total_context_tokens,
                    maximum_context_tokens,
                }
            }) if *actual_total_context_tokens == 5_000 && *maximum_context_tokens == 2_048
        ),
        "a worker-side embeddings failure must surface its typed reason: {embed_outcome:?}"
    );
    assert!(
        format!("{embed_outcome:?}").contains("context"),
        "the failure message should explain the context overflow: {embed_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
