//! The `astronomical embed` one-shot journey: text, file, or stdin in, one
//! JSON vector document on stdout, process exits. The daemon side is a stub
//! speaking the real framed protocol over a real unix socket.

use std::{
    io::{Read, Write},
    path::PathBuf,
};

use astronomical_cli::errors::EmbedError;
use astronomical_cli::{CliCommand, EmbedArguments, EmbedDependencies, UsageError, run_embed};
use astronomical_ipc_protocol::EmbeddingsFailureReason;
use tokio::time::timeout;

use super::stub_daemon::{
    StubDaemonConfig, StubEmbeddingsOutcome, StubInstalledModel, spawn_stub_daemon,
};
use super::test_support::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory, parse,
};

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
    stderr: &'a mut Vec<u8>,
) -> EmbedDependencies<'a> {
    EmbedDependencies {
        candidate_socket_paths,
        stdin,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
        download_stage_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    }
}

/// Stub with one resident embeddings model named as the effective default:
/// the no-flag resolution path.
fn resident_default_stub_config(embeddings_outcome: StubEmbeddingsOutcome) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::embeddings("test/local-embedder", true)],
        default_model_id: Some("test/local-embedder".to_owned()),
        embeddings_outcome,
        ..Default::default()
    }
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
    let test_directory = fresh_test_directory("embed", "text-argument");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        resident_default_stub_config(StubEmbeddingsOutcome::Completed),
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
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
    let test_directory = fresh_test_directory("embed", "file-input");
    let input_file_path = test_directory.join("input.txt");
    std::fs::write(&input_file_path, "hello from a file")
        .expect("the embed input file should be writable");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        resident_default_stub_config(StubEmbeddingsOutcome::Completed),
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
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
    let test_directory = fresh_test_directory("embed", "stdin-input");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        resident_default_stub_config(StubEmbeddingsOutcome::Completed),
    );
    let mut stdin_content = std::io::Cursor::new(b"hello from stdin".to_vec());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
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
    let test_directory = fresh_test_directory("embed", "not-running");
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![test_directory.join(SOCKET_FILE_NAME)],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
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
async fn should_reject_embed_when_no_model_is_available() {
    let test_directory = fresh_test_directory("embed", "no-model");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    // Nothing installed, no configured default, and the release catalog
    // offers no embeddings model: there is nothing to serve this request.
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            ..Default::default()
        },
    );
    let mut stdin_content = std::io::Cursor::new(b"hello".to_vec());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
        run_embed(&embed_arguments(None, None, None), &mut embed_dependencies),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    assert!(
        matches!(&embed_outcome, Err(EmbedError::ModelUnavailable { reason })
            if reason.contains("astronomical models supported")),
        "no embeddings model anywhere must fail with the pointer to the catalog: {embed_outcome:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_refuse_a_chat_only_model_for_embed() {
    let test_directory = fresh_test_directory("embed", "capmismatch");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    // The resident model chats but cannot embed: the capability pre-flight
    // must reject before any embeddings request goes out.
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            default_model_id: Some("test/local-chatter".to_owned()),
            ..Default::default()
        },
    );
    let mut stdin_content = std::io::Cursor::new(b"hello".to_vec());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
        run_embed(&embed_arguments(None, None, None), &mut embed_dependencies),
    )
    .await
    .expect("the embed journey should finish inside the test timeout");
    assert!(
        matches!(&embed_outcome, Err(EmbedError::ModelUnavailable { reason })
            if reason.contains("not an embeddings model")),
        "an embed request against a chat-only model must fail on capability: {embed_outcome:?}"
    );
    assert!(
        stdout.is_empty(),
        "no vector document may appear on a capability rejection"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_fail_with_the_worker_reason_when_embeddings_fail() {
    let test_directory = fresh_test_directory("embed", "worker-failure");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        resident_default_stub_config(StubEmbeddingsOutcome::ContextLengthExceeded),
    );
    let mut stdin_content = std::io::Cursor::new(Vec::new());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut embed_dependencies = embed_dependencies(
        vec![socket_path],
        &mut stdin_content,
        &mut stdout,
        &mut stderr,
    );

    let embed_outcome = timeout(
        TEST_TIMEOUT,
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
