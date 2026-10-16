use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;

use super::*;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("outer failure: {inner}")]
struct ChainedStartupError {
    #[source]
    inner: InnerStartupError,
}

#[derive(Debug, Error)]
#[error("inner root cause (socket EADDRINUSE)")]
struct InnerStartupError;

#[test]
fn should_record_the_full_cause_chain_in_the_failure_file() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    record_startup_failure(
        state_directory.path(),
        StartupFailureStage::InstanceLock,
        &ChainedStartupError {
            inner: InnerStartupError,
        },
        Some("pid=4242 instance=astronomicald".to_owned()),
    );

    let record = read_last_startup_failure(state_directory.path())
        .expect("the startup-failure record should be persisted");
    assert_eq!(record.stage, StartupFailureStage::InstanceLock);
    assert_eq!(
        record.message,
        "outer failure: inner root cause (socket EADDRINUSE)"
    );
    assert_eq!(
        record.cause_chain,
        vec![
            "outer failure: inner root cause (socket EADDRINUSE)".to_owned(),
            "inner root cause (socket EADDRINUSE)".to_owned(),
        ]
    );
    assert_eq!(
        record.context.as_deref(),
        Some("pid=4242 instance=astronomicald")
    );
    assert!(record.timestamp_unix_millis > 1_700_000_000_000);
}

#[test]
fn should_append_human_readable_lines_to_the_startup_failure_log() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    record_startup_failure(
        state_directory.path(),
        StartupFailureStage::Configuration,
        &ChainedStartupError {
            inner: InnerStartupError,
        },
        None,
    );

    let log_contents = std::fs::read_to_string(
        state_directory
            .path()
            .join("logs")
            .join("startup-failures.log"),
    )
    .expect("the startup-failure log should exist");
    assert!(log_contents.contains("[configuration] outer failure"));
    assert!(log_contents.contains("caused by inner root cause"));
}

#[test]
fn should_overwrite_the_previous_record_atomically_on_each_failure() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    record_startup_failure(
        state_directory.path(),
        StartupFailureStage::InstanceLock,
        &ChainedStartupError {
            inner: InnerStartupError,
        },
        None,
    );
    record_startup_failure(
        state_directory.path(),
        StartupFailureStage::LoggingInitialization,
        &ChainedStartupError {
            inner: InnerStartupError,
        },
        None,
    );

    let record = read_last_startup_failure(state_directory.path())
        .expect("the latest record should be readable");
    assert_eq!(record.stage, StartupFailureStage::LoggingInitialization);
    assert!(
        !state_directory
            .path()
            .join("last-startup-failure.json.tmp")
            .exists()
    );
}

#[test]
fn should_report_no_record_for_a_clean_state_directory() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    assert!(read_last_startup_failure(state_directory.path()).is_none());
}

#[test]
fn should_expose_the_lock_holder_note_written_by_the_running_daemon() {
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    assert_eq!(read_instance_lock_holder(state_directory.path()), None);

    let mut lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(state_directory.path().join(INSTANCE_LOCK_FILE_NAME))
        .expect("lock file fixture");
    use std::io::Write as _;
    lock_file
        .write_all(b"pid=99 instance=astronomicald\n")
        .expect("holder note write");

    assert_eq!(
        read_instance_lock_holder(state_directory.path()).as_deref(),
        Some("pid=99 instance=astronomicald")
    );
}

#[test]
fn should_not_leak_the_lock_file_note_across_releases() {
    // The holder note is advisory; an empty file (pre-upgrade daemons) must
    // read as "no note" rather than an empty holder string.
    let state_directory = tempfile::tempdir().expect("state directory fixture");
    File::create(state_directory.path().join(INSTANCE_LOCK_FILE_NAME))
        .expect("empty lock file fixture");
    assert_eq!(read_instance_lock_holder(state_directory.path()), None);
}
