//! The `astronomical unload` verb over the real framed protocol: the daemon
//! side is the shared stub, answering each unload outcome in turn.

use std::{io::Write, path::PathBuf};

use astronomical_cli::{UnloadDependencies, UnloadError};
use config::AstronomicalRuntimeInstance;
use ipc_protocol::DaemonModelUnloadStatus;
use tokio::time::timeout;

use crate::integration::support::environment;
use crate::integration::support::environment::{SOCKET_FILE_NAME, TEST_TIMEOUT};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubInstalledModel, spawn_stub_daemon,
};

fn unload_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
) -> UnloadDependencies<'a> {
    UnloadDependencies {
        candidate_socket_paths,
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdout: stdout as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
    }
}

/// Runs one unload against a stub answering `model_unload_status`.
async fn unload_against_stub(
    test_name: &str,
    model_unload_status: DaemonModelUnloadStatus,
) -> (Result<(), UnloadError>, String) {
    let test_directory = environment::fresh_test_directory("unload", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/resident-model", true)],
            model_unload_status,
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut dependencies = unload_dependencies(vec![socket_path], &mut stdout);
    let unload_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_unload(&mut dependencies),
    )
    .await
    .expect("the unload journey should finish inside the test timeout");
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    (
        unload_outcome,
        String::from_utf8(stdout).expect("the output is text"),
    )
}

#[tokio::test]
async fn should_unload_the_resident_model_and_name_it() {
    let (unload_outcome, printed_output) =
        unload_against_stub("unloaded", DaemonModelUnloadStatus::Unloaded).await;
    unload_outcome.expect("an unload of an idle model succeeds");
    assert_eq!(
        printed_output,
        "Unloaded test/resident-model. Its memory is free.\n"
    );
}

#[tokio::test]
async fn should_refuse_while_a_request_holds_the_model() {
    let (unload_outcome, printed_output) =
        unload_against_stub("busy", DaemonModelUnloadStatus::Busy).await;
    assert!(matches!(unload_outcome, Err(UnloadError::Busy)));
    assert!(printed_output.is_empty(), "{printed_output}");
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running() {
    let test_directory = environment::fresh_test_directory("unload", "no-daemon");
    let mut stdout = Vec::new();
    let mut dependencies =
        unload_dependencies(vec![test_directory.join(SOCKET_FILE_NAME)], &mut stdout);
    let unload_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_unload(&mut dependencies),
    )
    .await
    .expect("a missing daemon is reported promptly");
    assert!(matches!(
        unload_outcome,
        Err(UnloadError::DaemonNotRunning { .. })
    ));
    let _ = std::fs::remove_dir_all(&test_directory);
}
