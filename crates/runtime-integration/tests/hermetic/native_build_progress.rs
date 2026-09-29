//! Contracts for the live native-build progress stream that hosted CI tails
//! while cargo keeps the build script's own output captured.

use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

#[path = "../../build_progress.rs"]
mod build_progress;

use build_progress::NativeBuildProgress;

fn write_progress_file_path(progress_directory: &tempfile::TempDir) -> PathBuf {
    progress_directory.path().join("native-build-progress")
}

#[test]
fn should_write_operation_lifecycle_lines_to_the_progress_file() {
    let progress_directory =
        tempfile::tempdir().expect("the test should create a progress directory");
    let progress_file_path = write_progress_file_path(&progress_directory);
    let native_build_progress = NativeBuildProgress::from_file_path(progress_file_path.clone())
        .expect("an absolute progress path should be accepted");

    native_build_progress.record_build_start("15");
    let configure_progress = native_build_progress.begin_operation("native-configure");
    configure_progress.complete("success");
    let compile_progress = native_build_progress.begin_operation("native-compile");
    compile_progress.complete("failed");
    native_build_progress.record_build_completion(true, Duration::from_millis(1250));

    let progress_text =
        fs::read_to_string(&progress_file_path).expect("the test should read the progress file");
    assert!(
        progress_text.contains("[native-build-progress] status=start parallel_jobs=15"),
        "the stream should open with a build start line carrying the parallel job count: {progress_text}"
    );
    assert!(
        progress_text.contains("operation=native-configure status=start"),
        "each operation should announce its start: {progress_text}"
    );
    assert!(
        progress_text.contains("operation=native-configure status=success elapsed_seconds="),
        "each operation should announce its outcome with elapsed time: {progress_text}"
    );
    assert!(
        progress_text.contains("operation=native-compile status=failed elapsed_seconds="),
        "a failed operation should be recorded so CI sees where the build died: {progress_text}"
    );
    assert!(
        progress_text.contains("status=complete outcome=built elapsed_seconds="),
        "the stream should close with the overall build outcome: {progress_text}"
    );
}

#[test]
fn should_emit_heartbeat_lines_while_an_operation_is_running() {
    let progress_directory =
        tempfile::tempdir().expect("the test should create a progress directory");
    let progress_file_path = write_progress_file_path(&progress_directory);
    let native_build_progress = NativeBuildProgress::from_file_path(progress_file_path.clone())
        .expect("an absolute progress path should be accepted")
        .with_heartbeat_interval(Duration::from_millis(100));

    let compile_progress = native_build_progress.begin_operation("native-compile");
    thread::sleep(Duration::from_millis(350));
    compile_progress.complete("success");

    let progress_text =
        fs::read_to_string(&progress_file_path).expect("the test should read the progress file");
    let heartbeat_line_count = progress_text
        .lines()
        .filter(|line| line.contains("status=running"))
        .count();
    assert!(
        heartbeat_line_count >= 2,
        "a long operation should emit periodic heartbeats, found {heartbeat_line_count}: {progress_text}"
    );
}

#[test]
fn should_stop_heartbeat_lines_after_operation_completion() {
    let progress_directory =
        tempfile::tempdir().expect("the test should create a progress directory");
    let progress_file_path = write_progress_file_path(&progress_directory);
    let native_build_progress = NativeBuildProgress::from_file_path(progress_file_path.clone())
        .expect("an absolute progress path should be accepted")
        .with_heartbeat_interval(Duration::from_millis(50));

    let compile_progress = native_build_progress.begin_operation("native-compile");
    thread::sleep(Duration::from_millis(150));
    compile_progress.complete("success");
    let line_count_after_completion = fs::read_to_string(&progress_file_path)
        .expect("the test should read the progress file")
        .lines()
        .count();

    thread::sleep(Duration::from_millis(200));
    let line_count_after_quiet_period = fs::read_to_string(&progress_file_path)
        .expect("the test should read the progress file")
        .lines()
        .count();

    assert_eq!(
        line_count_after_completion, line_count_after_quiet_period,
        "no heartbeat may outlive its operation"
    );
}

#[test]
fn should_ignore_unwritable_progress_paths_instead_of_failing_the_build() {
    let progress_directory =
        tempfile::tempdir().expect("the test should create a progress directory");
    let unwritable_progress_path = progress_directory
        .path()
        .join("missing-directory")
        .join("native-build-progress");
    let native_build_progress =
        NativeBuildProgress::from_file_path(unwritable_progress_path.clone())
            .expect("an absolute progress path should be accepted");

    let configure_progress = native_build_progress.begin_operation("native-configure");
    configure_progress.complete("success");
    native_build_progress.record_build_completion(false, Duration::from_millis(5));

    assert!(
        !unwritable_progress_path.exists(),
        "an unwritable progress path must stay silent instead of creating partial state"
    );
}

#[test]
fn should_refuse_a_relative_progress_file_path() {
    let from_file_path_result =
        NativeBuildProgress::from_file_path(PathBuf::from("relative/progress-file"));

    assert!(
        from_file_path_result.is_err(),
        "a relative progress path would silently disable the stream in CI and must be rejected"
    );
}

#[test]
fn should_write_nothing_when_progress_is_disabled() {
    let native_build_progress = NativeBuildProgress::disabled();

    let configure_progress = native_build_progress.begin_operation("native-configure");
    configure_progress.complete("success");
    native_build_progress.record_build_completion(true, Duration::from_millis(1));
}
