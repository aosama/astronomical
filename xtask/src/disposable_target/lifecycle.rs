//! The supervision half of the disposable-target lifecycle: spawning the
//! journey command in its own process group, polling the self-pipe for
//! HUP/INT/TERM, stopping the group with grace then escalation, and the
//! exit-code precedence between child status and cleanup outcome. The
//! cleanup phases live in `cleanup`; entry point and pure helpers live in
//! the parent module.

#![allow(unsafe_code)]
// The workspace denies unsafe code everywhere else; the poll(2)/read(2)
// calls on the signal pipe below are the sanctioned edge for this module
// (see signal_edge.rs for the handler side).

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, ExitStatus};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use super::cleanup::ProgressReporter;
use super::signal_edge;
use super::{
    DISPOSABLE_REPOSITORY_ROOT_ENV, LaneJourney, command_is_available, exit_code_from_status,
    lifecycle_line, local_timestamp, print_error, select_lane_root, select_rustc_wrapper,
    unix_epoch_seconds,
};

const CHILD_TERMINATION_GRACE_SECONDS: u64 = 3;
const KILL_CONFIRMATION_SECONDS: u64 = 1;
const POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CleanupOperationName {
    Handoff,
    Measuring,
    Removing,
}

impl CleanupOperationName {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            CleanupOperationName::Handoff => "handoff",
            CleanupOperationName::Measuring => "measuring",
            CleanupOperationName::Removing => "removing",
        }
    }
}

pub(super) struct DisposableJourney {
    pub(super) lane_name: String,
    repository_root: PathBuf,
    pub(super) target_directory: PathBuf,
    selected_rustc_wrapper: Option<String>,
    pub(super) signal_pipe_read_fd: i32,
    pub(super) cleanup_operation: CleanupOperationName,
    pub(super) removal_result_receiver: Option<Receiver<std::io::Result<()>>>,
    pub(super) cleanup_progress_reporter: Option<ProgressReporter>,
    pub(super) is_cleanup_pending: bool,
}

pub(super) fn run_disposable_command(repository_root: &Path, journey: LaneJourney) -> ExitCode {
    let Some(command_environment) = select_command_environment() else {
        return ExitCode::from(1);
    };
    let Some(target_directory) =
        create_disposable_target(&command_environment.lane_root, &journey.lane_name)
    else {
        return ExitCode::from(1);
    };
    let Some(signal_pipe) = signal_edge::SignalPipe::install() else {
        print_error("could not install the disposable lifecycle signal handlers");
        return ExitCode::from(1);
    };

    let mut disposable_journey = DisposableJourney {
        lane_name: journey.lane_name.clone(),
        repository_root: repository_root.to_path_buf(),
        target_directory: target_directory.clone(),
        selected_rustc_wrapper: command_environment.rustc_wrapper,
        signal_pipe_read_fd: signal_pipe.read_fd,
        cleanup_operation: CleanupOperationName::Handoff,
        removal_result_receiver: None,
        cleanup_progress_reporter: None,
        is_cleanup_pending: true,
    };
    disposable_journey.log_start();

    let mut child_command = disposable_journey.build_child_command(&journey.command);
    let mut child_process = match child_command.spawn() {
        Ok(child_process) => child_process,
        Err(launch_error) => {
            print_error(&format!("could not run disposable command: {launch_error}"));
            let _ = disposable_journey.cleanup_owned_target();
            return ExitCode::from(1);
        }
    };
    let child_process_group_id = child_process.id() as i32;

    let journey_started_at_seconds = unix_epoch_seconds();
    let child_exit_status =
        match disposable_journey.await_child_exit(child_process_group_id, &mut child_process) {
            Ok(child_exit_status) => child_exit_status,
            Err(early_exit_code) => return early_exit_code,
        };

    let process_group_stop_failed = !disposable_journey.stop_child_process_group(
        "command-finished",
        child_process_group_id,
        &mut child_process,
    );
    if let Some(received_signal) = disposable_journey.drain_received_signal() {
        return disposable_journey.handle_received_signal_during_child(
            received_signal,
            child_process_group_id,
            &mut child_process,
        );
    }
    disposable_journey.log_command_finished(&child_exit_status, journey_started_at_seconds);
    if process_group_stop_failed {
        disposable_journey.skip_cleanup_for_active_process_group();
        return ExitCode::from(1);
    }

    let cleanup_succeeded = disposable_journey.cleanup_owned_target().is_ok();
    exit_code_after_cleanup(&child_exit_status, cleanup_succeeded)
}

/// The lane root and rustc wrapper resolved once and consumed by every later
/// phase of the disposable journey.
struct CommandEnvironment {
    lane_root: PathBuf,
    rustc_wrapper: Option<String>,
}

fn select_command_environment() -> Option<CommandEnvironment> {
    let selected_lane_root = match select_lane_root() {
        Ok(selected_lane_root) => selected_lane_root,
        Err(lane_root_error) => {
            print_error(&lane_root_error);
            return None;
        }
    };
    let configured_rustc_wrapper = std::env::var("RUSTC_WRAPPER")
        .ok()
        .filter(|value| !value.is_empty());
    if let Some(wrapper_name) = configured_rustc_wrapper.as_deref()
        && !command_is_available(wrapper_name)
    {
        print_error(&format!("required command is unavailable: {wrapper_name}"));
        return None;
    }
    let selected_rustc_wrapper = select_rustc_wrapper(
        configured_rustc_wrapper.as_deref(),
        command_is_available("sccache"),
    );
    Some(CommandEnvironment {
        lane_root: selected_lane_root,
        rustc_wrapper: selected_rustc_wrapper,
    })
}

fn create_disposable_target(lane_root: &Path, lane_name: &str) -> Option<PathBuf> {
    match super::create_marker_owned_target(lane_root, lane_name) {
        Ok(target_directory) => Some(target_directory),
        Err(creation_error) => {
            print_error(&format!(
                "could not create disposable Cargo target in {}: {creation_error}",
                lane_root.display()
            ));
            None
        }
    }
}

/// The exit-code precedence after supervision ends: the child's own status
/// outranks a cleanup failure, and a cleanup failure outranks success.
fn exit_code_after_cleanup(child_exit_status: &ExitStatus, cleanup_succeeded: bool) -> ExitCode {
    if !child_exit_status.success() {
        return ExitCode::from(exit_code_from_status(child_exit_status) as u8);
    }
    if !cleanup_succeeded {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

impl DisposableJourney {
    pub(super) fn poll_received_signal(&self) -> Option<i32> {
        let mut poll_descriptor = libc::pollfd {
            fd: self.signal_pipe_read_fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ready_count = unsafe {
            libc::poll(
                &mut poll_descriptor,
                1,
                POLL_INTERVAL.as_millis() as libc::c_int,
            )
        };
        if ready_count <= 0 {
            return None;
        }
        self.drain_received_signal()
    }

    pub(super) fn drain_received_signal(&self) -> Option<i32> {
        let mut signal_byte = [0u8; 1];
        let bytes_read = unsafe {
            libc::read(
                self.signal_pipe_read_fd,
                signal_byte.as_mut_ptr() as *mut libc::c_void,
                1,
            )
        };
        if bytes_read == 1 {
            Some(match signal_byte[0] as i32 {
                libc::SIGHUP => libc::SIGHUP,
                libc::SIGINT => libc::SIGINT,
                _ => libc::SIGTERM,
            })
        } else {
            None
        }
    }

    fn log_start(&self) {
        lifecycle_line(
            &self.lane_name,
            &format!(
                "start target_directory={} rustc_wrapper={} started_at={}",
                self.target_directory.display(),
                self.selected_rustc_wrapper.as_deref().unwrap_or("none"),
                local_timestamp()
            ),
            false,
        );
    }

    /// A dedicated process group lets interruption stop nested Cargo and
    /// native build children before their target disappears.
    fn build_child_command(&self, journey_command: &[String]) -> Command {
        let mut child_command = Command::new(&journey_command[0]);
        child_command
            .args(&journey_command[1..])
            .current_dir(&self.repository_root)
            .env("CARGO_TARGET_DIR", &self.target_directory)
            .env("ASTRONOMICAL_CARGO_TARGET_LIFECYCLE", "disposable")
            .env("ASTRONOMICAL_CARGO_TARGET_LANE", &self.lane_name)
            .env(
                DISPOSABLE_REPOSITORY_ROOT_ENV,
                self.repository_root
                    .canonicalize()
                    .unwrap_or_else(|_| self.repository_root.clone()),
            )
            .process_group(0);
        match self.selected_rustc_wrapper.as_deref() {
            Some(wrapper_name) => {
                child_command.env("RUSTC_WRAPPER", wrapper_name);
            }
            None => {
                child_command.env_remove("RUSTC_WRAPPER");
            }
        }
        child_command
    }

    /// Polls for signals and child exit until the child terminates; the
    /// error variant carries the exit code the supervisor must return early.
    fn await_child_exit(
        &mut self,
        child_process_group_id: i32,
        child_process: &mut Child,
    ) -> Result<ExitStatus, ExitCode> {
        loop {
            if let Some(received_signal) = self.poll_received_signal() {
                return Err(self.handle_received_signal_during_child(
                    received_signal,
                    child_process_group_id,
                    child_process,
                ));
            }
            match child_process.try_wait() {
                Ok(Some(child_exit_status)) => return Ok(child_exit_status),
                Ok(None) => continue,
                Err(wait_error) => {
                    print_error(&format!(
                        "could not observe the disposable command: {wait_error}"
                    ));
                    return Err(ExitCode::from(1));
                }
            }
        }
    }

    fn log_command_finished(
        &self,
        child_exit_status: &ExitStatus,
        journey_started_at_seconds: u64,
    ) {
        lifecycle_line(
            &self.lane_name,
            &format!(
                "command-finished exit_status={} elapsed_seconds={}",
                exit_code_from_status(child_exit_status),
                unix_epoch_seconds().saturating_sub(journey_started_at_seconds)
            ),
            false,
        );
    }

    /// A process group that survived the kill escalation may still hold the
    /// target's files open, so cleanup is abandoned and reported instead of
    /// racing the live group.
    fn skip_cleanup_for_active_process_group(&mut self) {
        self.is_cleanup_pending = false;
        lifecycle_line(
            &self.lane_name,
            &format!(
                "cleanup-skipped-active-process-group target_directory={}",
                self.target_directory.display()
            ),
            true,
        );
    }

    fn handle_received_signal_during_child(
        &mut self,
        received_signal: i32,
        child_process_group_id: i32,
        child_process: &mut Child,
    ) -> ExitCode {
        let signal_name = signal_edge::signal_name(received_signal);
        let signal_exit_status = signal_edge::signal_exit_status(received_signal);
        lifecycle_line(
            &self.lane_name,
            &format!("interrupted signal={signal_name}"),
            true,
        );
        if !self.stop_child_process_group(
            &format!("signal-{signal_name}"),
            child_process_group_id,
            child_process,
        ) {
            self.skip_cleanup_for_active_process_group();
            return ExitCode::from(signal_exit_status as u8);
        }
        let _ = self.cleanup_owned_target();
        ExitCode::from(signal_exit_status as u8)
    }

    /// Grace then escalation over the child's process group, returning
    /// false when the group stayed alive after both attempts.
    fn stop_child_process_group(
        &mut self,
        stop_reason: &str,
        process_group_id: i32,
        child_process: &mut Child,
    ) -> bool {
        if !signal_edge::process_group_is_alive(process_group_id) {
            let _ = child_process.wait();
            return true;
        }
        lifecycle_line(
            &self.lane_name,
            &format!(
                "stopping-process-group reason={stop_reason} grace_seconds={CHILD_TERMINATION_GRACE_SECONDS}"
            ),
            true,
        );
        signal_edge::signal_process_group(process_group_id, libc::SIGTERM);
        let mut grace_seconds_remaining = CHILD_TERMINATION_GRACE_SECONDS;
        while signal_edge::process_group_is_alive(process_group_id) && grace_seconds_remaining > 0 {
            std::thread::sleep(Duration::from_secs(1));
            grace_seconds_remaining -= 1;
        }
        if signal_edge::process_group_is_alive(process_group_id) {
            lifecycle_line(
                &self.lane_name,
                &format!("kill-escalation grace_seconds={CHILD_TERMINATION_GRACE_SECONDS}"),
                true,
            );
            signal_edge::signal_process_group(process_group_id, libc::SIGKILL);
            let mut kill_wait_seconds_remaining = KILL_CONFIRMATION_SECONDS;
            while signal_edge::process_group_is_alive(process_group_id)
                && kill_wait_seconds_remaining > 0
            {
                std::thread::sleep(Duration::from_secs(1));
                kill_wait_seconds_remaining -= 1;
            }
        }
        let _ = child_process.wait();
        if signal_edge::process_group_is_alive(process_group_id) {
            print_error(&format!(
                "child process group remained active after termination: {process_group_id}"
            ));
            return false;
        }
        true
    }
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod lifecycle_tests;
