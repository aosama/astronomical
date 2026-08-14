use std::time::Duration;

use tokio::time::timeout;

use super::daemon_process::{
    get_endpoint, post_empty_endpoint, spawn_actual_instance_daemon, terminate_daemon,
    write_instance_config,
};

#[tokio::test]
async fn should_keep_stable_running_while_development_starts_and_stops_independently() {
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("Cargo should provide the astronomicald executable path");
    let stable_state_directory = tempfile::tempdir().expect("stable state should be created");
    let development_state_directory =
        tempfile::tempdir().expect("development state should be created");
    write_instance_config(stable_state_directory.path(), "127.0.0.1:0");
    write_instance_config(development_state_directory.path(), "127.0.0.1:0");

    let (mut stable_daemon, stable_address) = spawn_actual_instance_daemon(
        &daemon_executable_path,
        "stable",
        stable_state_directory.path(),
    )
    .await;
    let stable_process_identifier = stable_daemon.id().expect("stable daemon should have a PID");
    let (mut development_daemon, development_address) = spawn_actual_instance_daemon(
        &daemon_executable_path,
        "development",
        development_state_directory.path(),
    )
    .await;

    let stable_status = get_endpoint(stable_address, "/v1/status").await;
    let development_status = get_endpoint(development_address, "/v1/status").await;
    assert!(stable_status.contains(r#""channel":"stable""#));
    assert!(development_status.contains(r#""channel":"development""#));
    assert!(stable_status.contains(r#""state_directory":"custom""#));
    assert!(development_status.contains(r#""state_directory":"custom""#));
    assert!(stable_status.contains(&format!(r#""version":"{}""#, env!("CARGO_PKG_VERSION"))));

    let development_shutdown =
        post_empty_endpoint(development_address, "/v1/control/shutdown").await;
    assert!(development_shutdown.starts_with("HTTP/1.1 202 Accepted"));
    assert!(
        timeout(Duration::from_secs(3), development_daemon.wait())
            .await
            .expect("development daemon should stop")
            .expect("development daemon should be reaped")
            .success()
    );

    assert_eq!(stable_daemon.id(), Some(stable_process_identifier));
    assert!(
        get_endpoint(stable_address, "/health")
            .await
            .starts_with("HTTP/1.1 200 OK")
    );
    terminate_daemon(&stable_daemon);
    assert!(
        timeout(Duration::from_secs(3), stable_daemon.wait())
            .await
            .expect("stable daemon should stop")
            .expect("stable daemon should be reaped")
            .success()
    );
}
