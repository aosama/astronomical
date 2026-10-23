//! The cleanup half of the disposable-target lifecycle: the rename-to-
//! `.removing` ownership handoff, the external killable `du -sk`
//! measurement with its five-second progress reporter, the worker-thread
//! removal, and the interruption semantics that retain exactly one
//! marker-owned target when a signal lands mid-cleanup. The struct and
//! supervision live in `lifecycle`; entry point and pure helpers live in
//! the parent module.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;
use std::time::Duration;

use super::lifecycle::{CleanupOperationName, DisposableJourney};
use super::signal_edge;
use super::{
    allocated_bytes_from_du_output, lifecycle_line, print_error, unix_epoch_seconds,
    validate_owned_target,
};

const PROGRESS_INTERVAL_SECONDS: u64 = 5;

pub(super) struct ProgressReporter {
    cancellation_flag: std::sync::Arc<Mutex<bool>>,
    reporter_thread: Option<JoinHandle<()>>,
}

impl ProgressReporter {
    fn start(lane_name: &str, operation_name: &'static str) -> ProgressReporter {
        let cancellation_flag = std::sync::Arc::new(Mutex::new(false));
        let thread_cancellation_flag = cancellation_flag.clone();
        let progress_lane_name = lane_name.to_owned();
        let reporter_thread = std::thread::spawn(move || {
            let mut elapsed_seconds = 0u64;
            loop {
                // Sleep in short slices so cancellation is prompt while the
                // printed counter still advances in fixed five-second steps,
                // matching the shell runner's interval reporter.
                for _ in 0..(PROGRESS_INTERVAL_SECONDS * 10) {
                    if *thread_cancellation_flag.lock().unwrap() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                elapsed_seconds += PROGRESS_INTERVAL_SECONDS;
                lifecycle_line(
                    &progress_lane_name,
                    &format!("{operation_name}-progress elapsed_seconds={elapsed_seconds}"),
                    true,
                );
            }
        });
        ProgressReporter {
            cancellation_flag,
            reporter_thread: Some(reporter_thread),
        }
    }

    fn stop(mut self) {
        *self.cancellation_flag.lock().unwrap() = true;
        if let Some(reporter_thread) = self.reporter_thread.take() {
            let _ = reporter_thread.join();
        }
    }
}

impl DisposableJourney {
    pub(super) fn cleanup_owned_target(&mut self) -> Result<(), ()> {
        if !self.is_cleanup_pending {
            return Ok(());
        }
        let cleanup_started_at_seconds = unix_epoch_seconds();
        if !self.target_directory.exists() && !self.target_directory.is_symlink() {
            self.is_cleanup_pending = false;
            lifecycle_line(&self.lane_name, "already-removed elapsed_seconds=0", false);
            return Ok(());
        }
        if let Err(refusal_message) = validate_owned_target(&self.target_directory, &self.lane_name)
        {
            self.is_cleanup_pending = false;
            print_error(&refusal_message);
            lifecycle_line(
                &self.lane_name,
                &format!(
                    "cleanup-refused target_directory={}",
                    self.target_directory.display()
                ),
                true,
            );
            return Err(());
        }
        let removal_target_directory = self.target_directory.with_file_name(format!(
            "{}.removing.{}",
            self.target_directory
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            std::process::id()
        ));
        if removal_target_directory.exists() || removal_target_directory.is_symlink() {
            self.is_cleanup_pending = false;
            print_error(&format!(
                "refusing occupied Cargo cleanup handoff path: {}",
                removal_target_directory.display()
            ));
            return Err(());
        }
        if let Err(handoff_error) =
            std::fs::rename(&self.target_directory, &removal_target_directory)
        {
            self.is_cleanup_pending = false;
            print_error(&format!(
                "failed to hand off owned Cargo target for cleanup: {}: {handoff_error}",
                self.target_directory.display()
            ));
            return Err(());
        }
        self.target_directory = removal_target_directory;
        if let Err(refusal_message) = validate_owned_target(&self.target_directory, &self.lane_name)
        {
            self.is_cleanup_pending = false;
            print_error(&refusal_message);
            lifecycle_line(
                &self.lane_name,
                &format!(
                    "cleanup-refused-after-handoff target_directory={}",
                    self.target_directory.display()
                ),
                true,
            );
            return Err(());
        }
        lifecycle_line(
            &self.lane_name,
            &format!(
                "cleanup-start target_directory={}",
                self.target_directory.display()
            ),
            false,
        );

        let measured_allocated_bytes = match self.measure_allocated_bytes() {
            Ok(measured_allocated_bytes) => measured_allocated_bytes,
            Err(()) => {
                self.is_cleanup_pending = false;
                return Err(());
            }
        };
        lifecycle_line(
            &self.lane_name,
            &format!("removing allocated_bytes={measured_allocated_bytes}"),
            false,
        );
        if !self.remove_owned_target() {
            self.is_cleanup_pending = false;
            return Err(());
        }
        self.is_cleanup_pending = false;
        lifecycle_line(
            &self.lane_name,
            &format!(
                "removed allocated_bytes={measured_allocated_bytes} elapsed_seconds={}",
                unix_epoch_seconds().saturating_sub(cleanup_started_at_seconds)
            ),
            false,
        );
        Ok(())
    }

    fn measure_allocated_bytes(&mut self) -> Result<u64, ()> {
        let spawned_measurement_child = Command::new("du")
            .arg("-sk")
            .arg(&self.target_directory)
            .stdout(Stdio::piped())
            .spawn();
        let mut measurement_child = match spawned_measurement_child {
            Ok(measurement_child) => measurement_child,
            Err(launch_error) => {
                print_error(&format!(
                    "could not measure allocated storage for {}: {launch_error}",
                    self.target_directory.display()
                ));
                return Err(());
            }
        };
        self.cleanup_operation = CleanupOperationName::Measuring;
        self.cleanup_progress_reporter =
            Some(ProgressReporter::start(&self.lane_name, "measuring"));

        let du_exit_status = loop {
            if let Some(received_signal) = self.poll_received_signal() {
                let signal_name = signal_edge::signal_name(received_signal);
                let signal_exit_status = signal_edge::signal_exit_status(received_signal);
                // Kill the measurement before reporting; the owned target
                // stays untouched on disk because measuring never mutates.
                let _ = measurement_child.kill();
                let _ = measurement_child.wait();
                self.finish_interrupted_cleanup(signal_name, signal_exit_status);
            }
            match measurement_child.try_wait() {
                Ok(Some(du_exit_status)) => break du_exit_status,
                Ok(None) => continue,
                Err(wait_error) => {
                    self.cleanup_operation = CleanupOperationName::Handoff;
                    if let Some(progress_reporter) = self.cleanup_progress_reporter.take() {
                        progress_reporter.stop();
                    }
                    print_error(&format!(
                        "could not measure allocated storage for {}: {wait_error}",
                        self.target_directory.display()
                    ));
                    return Err(());
                }
            }
        };
        self.cleanup_operation = CleanupOperationName::Handoff;
        if let Some(progress_reporter) = self.cleanup_progress_reporter.take() {
            progress_reporter.stop();
        }
        if !du_exit_status.success() {
            print_error(&format!(
                "could not measure allocated storage for {}",
                self.target_directory.display()
            ));
            return Err(());
        }
        let mut du_output = String::new();
        if let Some(mut du_standard_output) = measurement_child.stdout.take() {
            let _ = du_standard_output.read_to_string(&mut du_output);
        }
        match allocated_bytes_from_du_output(&du_output) {
            Ok(measured_allocated_bytes) => Ok(measured_allocated_bytes),
            Err(()) => {
                print_error(&format!(
                    "could not measure allocated storage for {}",
                    self.target_directory.display()
                ));
                Err(())
            }
        }
    }

    fn remove_owned_target(&mut self) -> bool {
        let removal_directory = self.target_directory.clone();
        let (removal_result_sender, removal_result_receiver) =
            std::sync::mpsc::channel::<std::io::Result<()>>();
        std::thread::spawn(move || {
            let removal_result = std::fs::remove_dir_all(&removal_directory);
            let _ = removal_result_sender.send(removal_result);
        });
        self.removal_result_receiver = Some(removal_result_receiver);
        self.cleanup_operation = CleanupOperationName::Removing;
        self.cleanup_progress_reporter = Some(ProgressReporter::start(&self.lane_name, "removing"));

        loop {
            if let Some(received_signal) = self.poll_received_signal() {
                let signal_name = signal_edge::signal_name(received_signal);
                let signal_exit_status = signal_edge::signal_exit_status(received_signal);
                self.finish_interrupted_cleanup(signal_name, signal_exit_status);
            }
            let removal_result = match self
                .removal_result_receiver
                .as_ref()
                .expect("the removing phase owns its result receiver")
                .try_recv()
            {
                Ok(removal_result) => removal_result,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => {
                    Err(std::io::Error::other("the removal worker died"))
                }
            };
            self.cleanup_operation = CleanupOperationName::Handoff;
            if let Some(progress_reporter) = self.cleanup_progress_reporter.take() {
                progress_reporter.stop();
            }
            if let Err(removal_error) = removal_result {
                print_error(&format!(
                    "failed to remove owned Cargo target: {}: {removal_error}",
                    self.target_directory.display()
                ));
                return false;
            }
            if self.target_directory.exists() || self.target_directory.is_symlink() {
                print_error(&format!(
                    "owned Cargo target remains after removal: {}",
                    self.target_directory.display()
                ));
                return false;
            }
            return true;
        }
    }

    /// Interruption reached while cleanup owns filesystem work: report the
    /// interrupted operation, then either retain the target (nothing was
    /// being removed) or wait for the active removal to finish and report
    /// which of the two terminal states it reached. Never starts a second
    /// recursive cleanup over active work.
    pub(super) fn finish_interrupted_cleanup(
        &mut self,
        signal_name: &str,
        signal_exit_status: i32,
    ) -> ! {
        lifecycle_line(
            &self.lane_name,
            &format!(
                "cleanup-interrupted signal={signal_name} operation={}",
                self.cleanup_operation.as_str()
            ),
            true,
        );
        // The measuring branch needs no work here: its caller kills the
        // external `du` child before entering, and measuring never mutates
        // the owned target, so retention is the correct terminal state.
        if self.cleanup_operation == CleanupOperationName::Removing {
            if let Some(removal_result_receiver) = self.removal_result_receiver.take() {
                let removal_result = removal_result_receiver
                    .recv()
                    .unwrap_or_else(|_| Err(std::io::Error::other("the removal worker died")));
                if let Some(progress_reporter) = self.cleanup_progress_reporter.take() {
                    progress_reporter.stop();
                }
                if removal_result.is_ok() && !self.target_directory.exists() {
                    lifecycle_line(&self.lane_name, "removed-during-interruption", false);
                } else {
                    lifecycle_line(
                        &self.lane_name,
                        &format!(
                            "cleanup-incomplete-retained target_directory={}",
                            self.target_directory.display()
                        ),
                        true,
                    );
                }
                self.is_cleanup_pending = false;
                std::process::exit(signal_exit_status);
            }
        }
        if let Some(progress_reporter) = self.cleanup_progress_reporter.take() {
            progress_reporter.stop();
        }
        lifecycle_line(
            &self.lane_name,
            &format!(
                "cleanup-skipped-retained target_directory={}",
                self.target_directory.display()
            ),
            true,
        );
        self.is_cleanup_pending = false;
        std::process::exit(signal_exit_status);
    }
}
