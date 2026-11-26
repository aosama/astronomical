//! What the journeys share when `respond` waits on a daemon that sends no frames: a
//! silence bound short enough to outlast, and a stub whose first frame comes late.

use std::time::{Duration, Instant};

use astronomical_cli::RespondError;
use tokio::time::timeout;

use super::respond_dependency_fixtures::respond_dependencies;
use crate::integration::support::environment::{
    DAEMON_WORK_POLL_INTERVAL, SILENCE_BOUND, SOCKET_FILE_NAME, TEST_TIMEOUT, fresh_test_directory,
    respond_arguments,
};
use crate::integration::support::stub_daemon::{
    StubDaemonConfig, StubInstalledModel, spawn_stub_daemon,
};

pub(super) fn stub_config_with_first_frame_delay(
    chat_first_frame_delay: Duration,
) -> StubDaemonConfig {
    StubDaemonConfig {
        installed_models: vec![StubInstalledModel::chat("test/local-chatter", true)],
        default_model_id: Some("test/local-chatter".to_owned()),
        chat_fragments: vec!["Hello".to_owned()],
        chat_first_frame_delay,
        ..Default::default()
    }
}

pub(super) struct RespondJourney {
    pub(super) respond_outcome: Result<(), RespondError>,
    pub(super) stdout: String,
    pub(super) stderr: String,
    pub(super) elapsed: Duration,
}

pub(super) async fn run_respond_against(
    test_name: &str,
    stub_config: StubDaemonConfig,
) -> RespondJourney {
    run_respond_with_stall_bound(test_name, stub_config, TEST_TIMEOUT).await
}

pub(super) async fn run_respond_with_stall_bound(
    test_name: &str,
    stub_config: StubDaemonConfig,
    model_load_stall_bound: Duration,
) -> RespondJourney {
    run_respond_with_bounds(
        test_name,
        stub_config,
        SILENCE_BOUND,
        model_load_stall_bound,
    )
    .await
}

pub(super) async fn run_respond_with_silence_bound(
    test_name: &str,
    stub_config: StubDaemonConfig,
    silence_bound: Duration,
) -> RespondJourney {
    run_respond_with_bounds(test_name, stub_config, silence_bound, TEST_TIMEOUT).await
}

async fn run_respond_with_bounds(
    test_name: &str,
    stub_config: StubDaemonConfig,
    silence_bound: Duration,
    model_load_stall_bound: Duration,
) -> RespondJourney {
    let test_directory = fresh_test_directory("wait", test_name);
    let socket_path = test_directory.join(SOCKET_FILE_NAME);
    let stub_daemon_task = spawn_stub_daemon(socket_path.clone(), stub_config);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut respond_dependencies =
        respond_dependencies(vec![socket_path], &mut stdout, &mut stderr);
    respond_dependencies.daemon_work_poll_interval = DAEMON_WORK_POLL_INTERVAL;
    respond_dependencies.request_timeout = silence_bound;
    respond_dependencies.model_load_stall_bound = model_load_stall_bound;

    let journey_started_at = Instant::now();
    let respond_outcome = timeout(
        TEST_TIMEOUT,
        astronomical_cli::run_respond(
            &respond_arguments("Say hello", None, false),
            &mut respond_dependencies,
        ),
    )
    .await
    .expect("the respond journey should finish inside the test timeout");
    let elapsed = journey_started_at.elapsed();

    stub_daemon_task.abort();
    let _ = std::fs::remove_dir_all(&test_directory);
    RespondJourney {
        respond_outcome,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        elapsed,
    }
}
