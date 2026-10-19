//! The `astronomical doctor` journey: one live channel behind a real stub
//! daemon and one dead channel, with the binding line naming the resolved
//! channel (issue #1391).

use astronomical_cli::DoctorDependencies;
use astronomical_config::AstronomicalRuntimeInstance;
use tokio::time::timeout;

use super::stub_daemon::{StubDaemonConfig, StubInstalledModel, spawn_stub_daemon};
use super::test_support::{SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory};

#[tokio::test]
async fn should_report_the_live_stubbed_channel_and_the_dead_channel() {
    let test_directory = fresh_test_directory("doctor", "mixed");
    let live_socket_path = test_directory.join(SOCKET_FILE_NAME);
    let dead_socket_path = test_directory.join("dead-development-daemon.sock");
    let stub_daemon_task = spawn_stub_daemon(
        live_socket_path.clone(),
        StubDaemonConfig {
            installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
            ..Default::default()
        },
    );
    let mut stdout = Vec::new();
    let mut doctor_dependencies = DoctorDependencies {
        executable_path: test_directory.join("astronomical"),
        channel_socket_paths: vec![
            (AstronomicalRuntimeInstance::Stable, live_socket_path),
            (AstronomicalRuntimeInstance::Development, dead_socket_path),
        ],
        stdout: &mut stdout,
        request_timeout: TEST_TIMEOUT,
    };

    let doctor_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_doctor(&mut doctor_dependencies),
    )
    .await
    .expect("the doctor journey should finish inside the test timeout");
    doctor_outcome.expect("the doctor journey should complete with mixed channel liveness");

    let rendered_stdout = String::from_utf8_lossy(&stdout);
    assert!(
        rendered_stdout.contains("binary:  "),
        "the report must lead with the binary path: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("status probe: live — daemon answered a status request"),
        "the stubbed stable daemon must report live: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("status probe: not running — socket file absent"),
        "the absent development socket must report not running: {rendered_stdout:?}"
    );
    assert!(
        rendered_stdout.contains("socket exists: yes")
            && rendered_stdout.contains("socket exists: no"),
        "one live and one dead channel must both state socket existence: {rendered_stdout:?}"
    );
    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
}
