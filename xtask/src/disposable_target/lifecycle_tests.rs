use std::os::unix::process::ExitStatusExt;
use std::process::{ExitCode, ExitStatus};

use super::exit_code_after_cleanup;

#[test]
fn should_return_success_when_the_child_succeeded_and_cleanup_succeeded() {
    let child_exit_status = ExitStatus::from_raw(0);
    assert_eq!(
        exit_code_after_cleanup(&child_exit_status, true),
        ExitCode::SUCCESS
    );
}

#[test]
fn should_return_the_child_exit_code_when_the_child_failed_despite_successful_cleanup() {
    let child_exit_status = ExitStatus::from_raw(3 << 8);
    assert_eq!(
        exit_code_after_cleanup(&child_exit_status, true),
        ExitCode::from(3)
    );
}

#[test]
fn should_return_failure_when_cleanup_failed_after_a_successful_child() {
    let child_exit_status = ExitStatus::from_raw(0);
    assert_eq!(
        exit_code_after_cleanup(&child_exit_status, false),
        ExitCode::from(1)
    );
}

#[test]
fn should_let_the_child_status_outrank_a_cleanup_failure() {
    let child_exit_status = ExitStatus::from_raw(7 << 8);
    assert_eq!(
        exit_code_after_cleanup(&child_exit_status, false),
        ExitCode::from(7)
    );
}

#[test]
fn should_map_a_signal_terminated_child_to_the_128_plus_signal_exit_code() {
    // Raw wait status 15 is a SIGTERM death; the shell convention the
    // lifecycle preserves reports that as 143.
    let child_exit_status = ExitStatus::from_raw(15);
    assert_eq!(
        exit_code_after_cleanup(&child_exit_status, true),
        ExitCode::from(143)
    );
}
