//! The `astronomical respond --schema` journey: schema validation, file
//! reading, and the schema reaching the daemon. These tests are split from
//! `respond_command.rs` to keep that file under the repo's size limit.

use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use astronomical_cli::RespondDependencies;
use astronomical_cli::errors::RespondError;
use astronomical_ipc_protocol::MAXIMUM_CHAT_SCHEMA_JSON_BYTES;
use tokio::time::timeout;

use super::stub_daemon;
use super::stub_daemon::{StubDaemonConfig, StubInstalledModel};
use super::test_support::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
    respond_arguments_with_schema,
};

fn respond_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
    stderr: &'a mut Vec<u8>,
) -> RespondDependencies<'a> {
    RespondDependencies {
        candidate_socket_paths,
        stdout: stdout as &mut dyn Write,
        stderr: stderr as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
        download_stage_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    }
}

fn resident_default_stub_config() -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
        default_model_id: Some("test/local-chatter".to_owned()),
        chat_fragments: vec!["Hello".to_owned(), " world".to_owned()],
        ..Default::default()
    }
}

#[tokio::test]
async fn should_send_the_schema_file_text_to_the_daemon() {
    let test_directory = fresh_test_directory("respond", "schema");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let schema_path = test_directory.join("answer-schema.json");
    let schema_text = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}";
    std::fs::write(&schema_path, schema_text).expect("the schema file should be writable");

    let schema_capture = Arc::new(Mutex::new(None));
    let mut stub_config = resident_default_stub_config();
    stub_config.schema_capture = schema_capture.clone();
    let stub_daemon_task = stub_daemon::spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_respond(
            &respond_arguments_with_schema("Reply as one JSON object", Some(schema_path)),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    respond_outcome.expect("the respond journey should complete against the stub daemon");

    let captured = schema_capture
        .lock()
        .expect("the schema capture lock should stay live")
        .take();
    assert_eq!(
        captured.as_deref(),
        Some(schema_text),
        "the schema text the user supplied must reach the daemon verbatim"
    );

    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

/// A missing schema file fails fast in the CLI, before any daemon contact:
/// the socket path below is unreachable on purpose to prove the ordering.
#[tokio::test]
async fn should_fail_before_connecting_when_the_schema_file_is_missing() {
    let test_directory = fresh_test_directory("respond", "schema-missing");
    let missing_schema_path = test_directory.join("absent-schema.json");
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies = respond_dependencies(
        vec![PathBuf::from("/nonexistent/ast-schema.sock")],
        &mut stdout,
        &mut stderr,
    );

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_respond(
            &respond_arguments_with_schema("Say hello", Some(missing_schema_path.clone())),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the schema read should finish inside the test timeout");
    match respond_outcome {
        Err(RespondError::SchemaReadFailed { path, .. }) => {
            assert_eq!(path, missing_schema_path)
        }
        other => panic!("expected a schema read failure, got {other:?}"),
    }
    assert!(stdout.is_empty(), "no answer text should reach stdout");

    let _ = std::fs::remove_dir_all(&test_directory);
}

/// The shared schema byte bound rejects an oversize schema file before any
/// daemon contact.
#[test]
fn should_reject_an_oversize_schema_file() {
    use astronomical_cli::respond_schema;

    let test_directory = fresh_test_directory("respond", "schema-oversize");
    let schema_path = test_directory.join("oversize-schema.json");
    let oversize_schema_text = format!(
        "{{\"values\":[\"{}\"]}}",
        "x".repeat(MAXIMUM_CHAT_SCHEMA_JSON_BYTES)
    );
    std::fs::write(&schema_path, oversize_schema_text)
        .expect("the oversize schema file should be writable");

    match respond_schema::read_schema_input(&schema_path) {
        Err(RespondError::SchemaTooLarge {
            actual_bytes,
            maximum_bytes,
        }) => {
            assert!(actual_bytes > MAXIMUM_CHAT_SCHEMA_JSON_BYTES);
            assert_eq!(maximum_bytes, MAXIMUM_CHAT_SCHEMA_JSON_BYTES);
        }
        other => panic!("expected a schema size failure, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&test_directory);
}

/// A schema file that is not valid UTF-8 text is rejected in the CLI with
/// the file path named, before any daemon contact.
#[test]
fn should_reject_a_non_utf8_schema_file() {
    use astronomical_cli::respond_schema;

    let test_directory = fresh_test_directory("respond", "schema-utf8");
    let schema_path = test_directory.join("binary-schema.json");
    std::fs::write(&schema_path, [0xFF, 0xFE, 0xFC])
        .expect("the binary schema file should be writable");

    match respond_schema::read_schema_input(&schema_path) {
        Err(RespondError::SchemaNotUtf8 { path, .. }) => assert_eq!(path, schema_path),
        other => panic!("expected a UTF-8 failure, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&test_directory);
}
