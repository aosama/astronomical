//! Lifecycle, heartbeat, and tolerance contracts for the native build progress
//! stream that CI tails into the live job log.

use std::{path::PathBuf, time::Duration};

#[path = "../../build_progress.rs"]
mod build_progress;

use build_progress::{NATIVE_BUILD_PROGRESS_FILE_VARIABLE, NativeBuildProgress};

const TEST_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(10);

fn progress_file_path(temporary_directory: &std::path::Path, case_name: &str) -> PathBuf {
    temporary_directory.join(format!("{case_name}-progress.log"))
}

fn read_progress_lines(progress_file_path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(progress_file_path)
        .expect("the progress file should be readable")
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn should_stream_lifecycle_events_for_a_successful_operation() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    let progress_file_path = progress_file_path(temporary_directory.path(), "lifecycle");
    let native_build_progress =
        NativeBuildProgress::new(Some(progress_file_path.clone()), TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> =
        native_build_progress.run_operation("native-compile", || Ok(()));

    assert_eq!(operation_result, Ok(()));
    let progress_lines = read_progress_lines(&progress_file_path);
    assert!(
        progress_lines
            .iter()
            .any(|progress_line| progress_line.contains("operation=native-compile status=start")),
        "the operation start should be streamed"
    );
    let success_line = progress_lines
        .iter()
        .find(|progress_line| progress_line.contains("operation=native-compile status=success"))
        .expect("the success outcome should be streamed");
    assert!(
        success_line.contains("elapsed_seconds="),
        "the success line should attribute elapsed time: {success_line}"
    );
}

#[test]
fn should_stream_the_failure_reason_for_a_failed_operation() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    let progress_file_path = progress_file_path(temporary_directory.path(), "failure");
    let native_build_progress =
        NativeBuildProgress::new(Some(progress_file_path.clone()), TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> = native_build_progress
        .run_operation("native-configure", || {
            Err("synthetic native failure".to_owned())
        });

    assert_eq!(operation_result.unwrap_err(), "synthetic native failure");
    let progress_lines = read_progress_lines(&progress_file_path);
    let failed_line = progress_lines
        .iter()
        .find(|progress_line| progress_line.contains("operation=native-configure status=failed"))
        .expect("the failure outcome should be streamed");
    assert!(
        failed_line.contains("error=synthetic native failure"),
        "the failed line should carry the failure reason: {failed_line}"
    );
    assert!(
        failed_line.contains("elapsed_seconds="),
        "the failed line should attribute elapsed time: {failed_line}"
    );
    assert!(
        !progress_lines
            .iter()
            .any(|progress_line| progress_line.contains("status=success")),
        "a failed operation must not stream a success outcome"
    );
}

#[test]
fn should_stream_heartbeats_while_a_long_operation_runs() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    let progress_file_path = progress_file_path(temporary_directory.path(), "heartbeat");
    let native_build_progress =
        NativeBuildProgress::new(Some(progress_file_path.clone()), TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> =
        native_build_progress.run_operation("native-compile", || {
            std::thread::sleep(Duration::from_millis(200));
            Ok(())
        });

    assert_eq!(operation_result, Ok(()));
    let progress_lines = read_progress_lines(&progress_file_path);
    let heartbeat_count = progress_lines
        .iter()
        .filter(|progress_line| progress_line.contains("status=heartbeat"))
        .count();
    assert!(
        heartbeat_count >= 2,
        "a long operation should stream periodic heartbeats, found {heartbeat_count}"
    );
    assert!(
        progress_lines
            .iter()
            .filter(|progress_line| progress_line.contains("status=heartbeat"))
            .all(|progress_line| progress_line.contains("operation=native-compile")),
        "heartbeats should name the operation they keep visible"
    );
}

#[test]
fn should_not_stream_heartbeats_after_the_operation_completes() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    let progress_file_path = progress_file_path(temporary_directory.path(), "heartbeat-order");
    let native_build_progress =
        NativeBuildProgress::new(Some(progress_file_path.clone()), TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> =
        native_build_progress.run_operation("native-compile", || {
            std::thread::sleep(Duration::from_millis(200));
            Ok(())
        });

    assert_eq!(operation_result, Ok(()));
    let progress_lines = read_progress_lines(&progress_file_path);
    let success_line_index = progress_lines
        .iter()
        .position(|progress_line| progress_line.contains("status=success"))
        .expect("the success outcome should be streamed");
    assert!(
        progress_lines[success_line_index + 1..]
            .iter()
            .all(|progress_line| !progress_line.contains("status=heartbeat")),
        "no heartbeat may appear after the operation outcome"
    );
}

#[test]
fn should_stream_recorded_events_to_the_progress_file() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    let progress_file_path = progress_file_path(temporary_directory.path(), "recorded");
    let native_build_progress =
        NativeBuildProgress::new(Some(progress_file_path.clone()), TEST_HEARTBEAT_INTERVAL);

    native_build_progress.record_event("status=started profile=core");

    let progress_lines = read_progress_lines(&progress_file_path);
    assert!(
        progress_lines
            .iter()
            .any(|progress_line| progress_line.contains("status=started profile=core")),
        "recorded lifecycle events should reach the progress file"
    );
}

#[test]
fn should_keep_building_when_the_progress_file_is_unwritable() {
    let temporary_directory = tempfile::tempdir().expect("the test should create temporary output");
    // A directory where the progress file should be makes every append fail.
    let unwritable_progress_path = temporary_directory.path().join("progress-directory");
    std::fs::create_dir(&unwritable_progress_path)
        .expect("the test should create the unwritable progress boundary");
    let native_build_progress =
        NativeBuildProgress::new(Some(unwritable_progress_path), TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> =
        native_build_progress.run_operation("native-compile", || Ok(()));

    assert_eq!(operation_result, Ok(()));
    native_build_progress.record_event("status=started profile=core");
}

#[test]
fn should_operate_without_a_progress_file() {
    let native_build_progress = NativeBuildProgress::new(None, TEST_HEARTBEAT_INTERVAL);

    let operation_result: Result<(), String> =
        native_build_progress.run_operation("native-compile", || Ok(()));

    assert_eq!(operation_result, Ok(()));
    native_build_progress.record_event("status=started profile=core");
}

#[test]
fn should_construct_from_the_ambient_environment() {
    let ambient_native_build_progress = NativeBuildProgress::from_environment();

    let operation_result: Result<(), String> =
        ambient_native_build_progress.run_operation("native-compile", || Ok(()));

    assert_eq!(operation_result, Ok(()));
}

#[test]
fn should_expose_the_progress_file_variable_to_cargo_rerun_contracts() {
    const NATIVE_BUILD_SCRIPT_SOURCE: &str = include_str!("../../build.rs");
    const PROGRESS_MODULE_SOURCE: &str = include_str!("../../build_progress.rs");

    assert_eq!(
        NATIVE_BUILD_PROGRESS_FILE_VARIABLE, "ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE",
        "the progress file variable name must stay stable for CI"
    );
    assert!(
        PROGRESS_MODULE_SOURCE.contains("ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE"),
        "the progress file variable name must stay stable for CI"
    );
    assert!(
        NATIVE_BUILD_SCRIPT_SOURCE.contains("NATIVE_BUILD_PROGRESS_FILE_VARIABLE,"),
        "changing the progress file must rerun the native build script"
    );
}
