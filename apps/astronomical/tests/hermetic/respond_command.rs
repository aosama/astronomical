//! The `astronomical respond` one-shot journey: prompt in, streamed answer
//! out, process exits. The daemon side is a stub speaking the real framed
//! protocol over a real unix socket.

use std::{io::Write, path::PathBuf};

use astronomical_cli::errors::RespondError;
use astronomical_cli::{
    CliCommand, RespondArguments, RespondDependencies, UsageError, run_respond,
};
use tokio::time::timeout;

use super::stub_daemon::{
    StubCatalogEntry, StubDaemonConfig, StubInstalledModel, spawn_stub_daemon, stub_download_job,
};
use super::test_support::{
    DOWNLOAD_POLL_INTERVAL, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory, parse,
};

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
        request_timeout: TEST_TIMEOUT,
        download_stage_bound: TEST_TIMEOUT,
        download_poll_interval: DOWNLOAD_POLL_INTERVAL,
    }
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
    let test_directory = fresh_test_directory("respond", "not-running");
    let missing_socket_path = test_directory.join(SOCKET_FILE_NAME);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![missing_socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
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

/// Stub that has one resident chat model and names it as the effective
/// default: the no-flag resolution path.
fn resident_default_stub_config() -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
        default_model_id: Some("test/local-chatter".to_owned()),
        chat_fragments: vec!["Hello".to_owned(), " world".to_owned()],
        ..Default::default()
    }
}

#[tokio::test]
async fn should_stream_generated_text_to_stdout() {
    let test_directory = fresh_test_directory("respond", "stream-text");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), resident_default_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
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
    let test_directory = fresh_test_directory("respond", "no-stream");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), resident_default_stub_config());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
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
async fn should_fail_when_the_requested_model_is_unknown() {
    let test_directory = fresh_test_directory("respond", "unknown-model");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/other-model", true)],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
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
            Err(RespondError::ModelUnavailable { reason })
                if reason.contains("test/wanted-model")
        ),
        "requesting a model the machine cannot serve must fail with the model id: {respond_outcome:?}"
    );
    let rendered_outcome = format!("{respond_outcome:?}");
    assert!(
        rendered_outcome.contains("did you mean: test/other-model"),
        "the rejection should list the near match the machine actually has: {rendered_outcome}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_auto_load_the_builtin_default_when_nothing_is_resident() {
    let test_directory = fresh_test_directory("respond", "auto-load-default");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    // Cold daemon: nothing installed or resident, but the built-in default
    // is on this Mac in the catalog, so the request must simply stream.
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![StubCatalogEntry::chat(
                "mlx-community/Qwen3.5-2B-4bit",
                "Qwen3.5-2B-4bit",
                true,
            )],
            chat_fragments: vec!["Loaded".to_owned(), " fine".to_owned()],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    respond_outcome
        .expect("a cold daemon with the built-in default on disk must serve the request");
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "Loaded fine",
        "the daemon loads the resident model itself; the CLI just streams"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_download_a_missing_model_before_streaming() {
    let test_directory = fresh_test_directory("respond", "auto-download");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    // The requested model is not on this Mac: the lifecycle starts the
    // download, waits for the catalog to flip to ready, then streams.
    let missing_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", false);
    let ready_entry = StubCatalogEntry::chat("test/downloaded-model", "downloaded-model", true);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![missing_entry.clone()],
            catalog_entries_after_download: Some(vec![ready_entry]),
            download_jobs: vec![
                Some(stub_download_job(
                    "test/downloaded-model",
                    "downloading",
                    1_000_000_000,
                    2_000_000_000,
                    None,
                )),
                None,
            ],
            chat_fragments: vec!["Downloaded".to_owned()],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", Some("test/downloaded-model"), false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    respond_outcome.expect("an auto-download must complete before the stream");
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "Downloaded",
        "the answer must stream only after the download finished"
    );
    let rendered_stderr = String::from_utf8_lossy(&stderr);
    assert!(
        rendered_stderr.contains("1 GB / 2 GB"),
        "the live progress should report decimal GB to stderr: {rendered_stderr:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_reject_a_chat_request_for_an_embeddings_only_model_before_downloading() {
    let test_directory = fresh_test_directory("respond", "capmismatch");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    // An embeddings-only catalog model that is not on this Mac yet: the
    // capability check must reject before any download work starts.
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            catalog_entries: vec![StubCatalogEntry::embeddings(
                "test/only-embeddings",
                "only-embeddings",
                false,
            )],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);

    let respond_outcome = timeout(
        TEST_TIMEOUT,
        run_respond(
            &respond_arguments("Say hello", Some("test/only-embeddings"), false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    assert!(
        matches!(
            &respond_outcome,
            Err(RespondError::ModelUnavailable { reason })
                if reason.contains("not a chat model")
        ),
        "a chat request against an embeddings-only model must fail on capability: {respond_outcome:?}"
    );
    assert!(
        stdout.is_empty(),
        "no answer may stream on a capability rejection"
    );
    let rendered_stderr = String::from_utf8_lossy(&stderr);
    assert!(
        !rendered_stderr.contains("downloading"),
        "no download may start for a capability mismatch: {rendered_stderr:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
