//! The `astronomical status` verb: worker state, effective default, and the
//! active download job. The daemon side is the shared stub speaking the real
//! framed protocol.

use std::{io::Write, path::PathBuf};

use astronomical_cli::StatusDependencies;
use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::{DaemonWorkerStatus, ModelLoadInFlight, NeuralEnginePrefillStatus};
use tokio::time::timeout;

use crate::integration::support::environment;
use crate::integration::support::environment::{SOCKET_FILE_NAME, TEST_TIMEOUT};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubInstalledModel, spawn_stub_daemon, stub_download_job,
};

fn status_dependencies<'a>(
    candidate_socket_paths: Vec<PathBuf>,
    stdout: &'a mut Vec<u8>,
) -> StatusDependencies<'a> {
    StatusDependencies {
        candidate_socket_paths,
        runtime_instance: AstronomicalRuntimeInstance::Development,
        stdout: stdout as &mut dyn Write,
        request_timeout: TEST_TIMEOUT,
    }
}

#[tokio::test]
async fn should_report_a_ready_worker_with_resident_model_and_default() {
    let test_directory = environment::fresh_test_directory("status", "ready");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            default_model_id: Some("test/local-chatter".to_owned()),
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("worker:   ready (resident: test/local-chatter)"),
        "a ready daemon with a resident model should say so: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("default:  test/local-chatter"),
        "the effective default should appear: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("download: none"),
        "with no active job the download line should be none: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_loading_worker_without_a_resident_model() {
    let test_directory = environment::fresh_test_directory("status", "loading");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            worker_status: DaemonWorkerStatus::Loading,
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("worker:   loading"),
        "a loading daemon should render the loading line: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("default:  none"),
        "with no configured default the line should be none: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_an_unavailable_worker() {
    let test_directory = environment::fresh_test_directory("status", "unavailable");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            worker_status: DaemonWorkerStatus::Unavailable,
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("worker:   unavailable"),
        "an unavailable worker should render the unavailable line: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_an_active_download_with_marketed_gigabytes() {
    let test_directory = environment::fresh_test_directory("status", "downloading");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            download_jobs: vec![Some(stub_download_job(
                "test/downloading-model",
                "downloading",
                500_000_000,
                2_000_000_000,
                None,
            ))],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout
            .contains("download: test/downloading-model — downloading 0.47 GB / 1.86 GB"),
        "an active job should render its state and marketed GB: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_failed_download_job() {
    let test_directory = environment::fresh_test_directory("status", "failed-download");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            download_jobs: vec![Some(stub_download_job(
                "test/broken-model",
                "failed",
                0,
                2_000_000_000,
                Some("network unreachable"),
            ))],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("download: test/broken-model (failed: network unreachable)"),
        "a failed job should render its error: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_report_a_missing_daemon_as_not_running_for_status() {
    let test_directory = environment::fresh_test_directory("status", "not-running");
    let mut stdout = Vec::new();
    let mut status_dependencies =
        status_dependencies(vec![test_directory.join(SOCKET_FILE_NAME)], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    assert!(
        matches!(
            status_outcome,
            Err(astronomical_cli::StatusError::DaemonNotRunning { .. })
        ),
        "a missing daemon must fail with the not-running error: {status_outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_show_a_load_in_flight_on_its_own_line() {
    let test_directory = environment::fresh_test_directory("status", "load-progress");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/old-model", true)],
            model_load_reports: vec![ModelLoadInFlight::for_model("test/new-model")],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("loading:  test/new-model: Loading the model"),
        "a load in flight should say what it is doing: {rendered_stdout:?}"
    );
    assert!(
        !rendered_stdout.contains("test/old-model"),
        "the model being replaced is no longer resident: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

#[tokio::test]
async fn should_show_no_loading_line_when_nothing_is_loading() {
    let test_directory = environment::fresh_test_directory("status", "no-load-progress");
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), StubDaemonConfig::default());
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);

    let status_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout");
    status_outcome.expect("the status journey should complete against the stub daemon");
    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        !rendered_stdout.contains("loading:"),
        "an idle status keeps its usual lines: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}

async fn status_report_with_neural_engine(
    test_name: &str,
    neural_engine_prefill: Option<NeuralEnginePrefillStatus>,
) -> String {
    let test_directory = environment::fresh_test_directory("status", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(
        socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/model", true)],
            neural_engine_prefill,
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut status_dependencies = status_dependencies(vec![socket_path], &mut stdout);
    timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_status(&mut status_dependencies),
    )
    .await
    .expect("the status journey should finish inside the test timeout")
    .expect("the status journey should complete against the stub daemon");
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    String::from_utf8(stdout).expect("the report is text")
}

#[tokio::test]
async fn should_show_programs_being_prepared_beside_the_serving_model() {
    let report = status_report_with_neural_engine(
        "ne-preparing",
        Some(NeuralEnginePrefillStatus::Preparing {
            completed_program_count: 12,
            total_program_count: 56,
            elapsed_ms: 12_000,
            estimated_remaining_ms: None,
        }),
    )
    .await;

    assert!(
        report.contains("neural engine:  preparing, 12/56 programs"),
        "{report}"
    );
}

#[tokio::test]
async fn should_add_no_neural_engine_line_when_the_setting_is_off() {
    let report_when_off =
        status_report_with_neural_engine("ne-off", Some(NeuralEnginePrefillStatus::Off)).await;
    let report_before_any_model = status_report_with_neural_engine("ne-none", None).await;

    assert!(
        !report_when_off.contains("neural engine"),
        "{report_when_off}"
    );
    assert!(
        !report_before_any_model.contains("neural engine"),
        "{report_before_any_model}"
    );
}
