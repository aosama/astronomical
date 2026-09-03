use std::{
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    time::{Duration, Instant},
};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ImageGenerationCommand, ProtocolReader, ProtocolWriter, RequestId,
    WorkerCommand, WorkerEvent, WorkerStartupConfiguration,
};
use tokio::{
    io::AsyncReadExt,
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    task::JoinHandle,
    time::timeout,
};

use crate::{WorkerControlError, worker_stderr_tail::WorkerStderrTail};

const WORKER_STDERR_READ_BYTES: usize = 1_024;
const WORKER_EXIT_DIAGNOSTIC_SETTLE_TIMEOUT: Duration = Duration::from_secs(1);
// A worker that just closed its stdout is milliseconds from exiting. The
// exit diagnostic waits this long for the natural exit before falling back
// to the "still running" snapshot, so the reported status is the real exit
// code even when the kernel has not reaped the child at the instant of the
// first check.
const WORKER_NATURAL_EXIT_GRACE_TIMEOUT: Duration = Duration::from_secs(1);

/// Outcome reported after a worker process is terminated and reaped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerTerminationOutcome {
    /// Closing worker input was sufficient to terminate and reap the process.
    Graceful {
        /// Whether the child reported a successful process exit.
        process_exit_successful: bool,
    },
    /// The supervisor had to send a termination signal before reaping.
    Forced {
        /// Whether the child nevertheless reported a successful process exit.
        process_exit_successful: bool,
    },
}

impl WorkerTerminationOutcome {
    /// Returns whether the underlying process exit status was successful.
    #[must_use]
    pub const fn was_successful(self) -> bool {
        match self {
            Self::Graceful {
                process_exit_successful,
            }
            | Self::Forced {
                process_exit_successful,
            } => process_exit_successful,
        }
    }
}

const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
const WORKER_COMMAND_WRITE_TIMEOUT: Duration = Duration::from_secs(1);

/// One local inference child process and its bounded stdio connection.
pub struct WorkerProcess {
    command_writer: Option<ProtocolWriter<ChildStdin>>,
    event_reader: ProtocolReader<ChildStdout>,
    command_write_timeout: Duration,
    launch_executable_path: PathBuf,
    launch_startup_configuration: Option<WorkerStartupConfiguration>,
    shutdown_timeout: Duration,
    stderr_drain_task: Option<JoinHandle<()>>,
    worker_stderr_tail: WorkerStderrTail,
    worker_started_at: Instant,
    worker_process: Child,
}

impl WorkerProcess {
    /// Starts the worker executable and captures its stdio streams.
    ///
    pub async fn launch(
        worker_executable_path: impl AsRef<Path>,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_timeouts(
            worker_executable_path,
            WORKER_COMMAND_WRITE_TIMEOUT,
            WORKER_SHUTDOWN_TIMEOUT,
        )
        .await
    }

    /// Starts a production worker and supplies its resolved startup configuration.
    pub async fn launch_with_startup_configuration(
        worker_executable_path: impl AsRef<Path>,
        worker_startup_configuration: WorkerStartupConfiguration,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_timeouts_and_startup_configuration(
            worker_executable_path,
            WORKER_COMMAND_WRITE_TIMEOUT,
            WORKER_SHUTDOWN_TIMEOUT,
            Some(worker_startup_configuration),
        )
        .await
    }

    /// Starts a worker with explicit command and shutdown timeouts.
    pub async fn launch_with_timeouts(
        worker_executable_path: impl AsRef<Path>,
        command_write_timeout: Duration,
        shutdown_timeout: Duration,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_timeouts_and_startup_configuration(
            worker_executable_path,
            command_write_timeout,
            shutdown_timeout,
            None,
        )
        .await
    }

    async fn launch_with_timeouts_and_startup_configuration(
        worker_executable_path: impl AsRef<Path>,
        command_write_timeout: Duration,
        shutdown_timeout: Duration,
        worker_startup_configuration: Option<WorkerStartupConfiguration>,
    ) -> Result<Self, WorkerControlError> {
        let launch_executable_path = worker_executable_path.as_ref().to_path_buf();
        let launch_startup_configuration = worker_startup_configuration.clone();
        let mut command = Command::new(&launch_executable_path);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let worker_launch_started_at = Instant::now();
        tracing::info!("starting local inference worker process");
        let mut worker_process = command.spawn().map_err(WorkerControlError::StartWorker)?;
        let worker_started_at = Instant::now();
        tracing::info!(
            worker_process_id = ?worker_process.id(),
            launch_elapsed_millis = worker_launch_started_at.elapsed().as_millis(),
            "started local inference worker process"
        );
        let worker_stderr_tail = WorkerStderrTail::default();
        let stderr_drain_task = match worker_process.stderr.take() {
            Some(worker_stderr) => {
                spawn_worker_stderr_drain(worker_stderr, worker_stderr_tail.clone())
            }
            None => {
                return Err(cleanup_startup_failure(
                    WorkerControlError::MissingStandardError,
                    &mut worker_process,
                    shutdown_timeout,
                )
                .await);
            }
        };
        let worker_stdin = match worker_process.stdin.take() {
            Some(worker_stdin) => worker_stdin,
            None => {
                return Err(cleanup_startup_failure(
                    WorkerControlError::MissingStandardInput,
                    &mut worker_process,
                    shutdown_timeout,
                )
                .await);
            }
        };
        let worker_stdout = match worker_process.stdout.take() {
            Some(worker_stdout) => worker_stdout,
            None => {
                return Err(cleanup_startup_failure(
                    WorkerControlError::MissingStandardOutput,
                    &mut worker_process,
                    shutdown_timeout,
                )
                .await);
            }
        };

        let mut command_writer = ProtocolWriter::new(worker_stdin);
        if let Some(worker_startup_configuration) = worker_startup_configuration {
            let initialization_write_result = timeout(
                command_write_timeout,
                command_writer.send_command(&WorkerCommand::InitializeWorker(
                    worker_startup_configuration,
                )),
            )
            .await
            .map_err(|_| WorkerControlError::CommandWriteTimeout {
                command_timeout_millis: command_write_timeout.as_millis(),
            })
            .and_then(|write_result| write_result.map_err(Into::into));
            if let Err(initialization_write_error) = initialization_write_result {
                return Err(cleanup_startup_failure(
                    initialization_write_error,
                    &mut worker_process,
                    shutdown_timeout,
                )
                .await);
            }
        }
        Ok(Self {
            command_writer: Some(command_writer),
            event_reader: ProtocolReader::new(worker_stdout),
            command_write_timeout,
            launch_executable_path,
            launch_startup_configuration,
            shutdown_timeout,
            stderr_drain_task: Some(stderr_drain_task),
            worker_stderr_tail,
            worker_started_at,
            worker_process,
        })
    }

    /// Starts a clean process from this worker's exact portable launch inputs.
    pub(crate) async fn relaunch_after_termination(&mut self) -> Result<(), WorkerControlError> {
        let replacement_worker = Self::launch_with_timeouts_and_startup_configuration(
            &self.launch_executable_path,
            self.command_write_timeout,
            self.shutdown_timeout,
            self.launch_startup_configuration.clone(),
        )
        .await?;
        *self = replacement_worker;
        Ok(())
    }

    pub(crate) fn expected_configuration_generation(&self) -> Option<&str> {
        self.launch_startup_configuration
            .as_ref()
            .map(|configuration| configuration.configuration_generation.as_str())
    }

    pub async fn start_generation(
        &mut self,
        generation_command: ChatGenerationCommand,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::Generate(generation_command))
            .await
    }

    pub async fn start_image_generation(
        &mut self,
        generation_command: ImageGenerationCommand,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::GenerateImage(generation_command))
            .await
    }

    pub async fn cancel_generation(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::Cancel { request_id })
            .await
    }

    /// Requests one non-critical MLX memory observation from an idle worker.
    pub async fn sample_mlx_memory(&mut self) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::SampleMlxMemory)
            .await
    }

    /// Requests one live MLX memory-ceiling update from an idle worker.
    pub async fn update_mlx_memory_limit(
        &mut self,
        effective_mlx_memory_ceiling_bytes: u64,
        configuration_generation: String,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes,
            configuration_generation,
        })
        .await
    }

    /// Requests a global or model-scoped persistent prompt-cache deletion.
    pub async fn clear_prompt_cache(
        &mut self,
        model_id: Option<String>,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::ClearPromptCache { model_id })
            .await
    }

    /// Sends a SwapModel command to the worker, instructing it to unload the current
    /// model and load a new one from the given directory.
    pub async fn swap_model(
        &mut self,
        model_directory: String,
        model_configuration: astronomical_ipc_protocol::WorkerModelConfiguration,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::SwapModel {
            model_directory,
            model_configuration,
        })
        .await
    }

    pub async fn next_event(&mut self) -> Result<Option<WorkerEvent>, WorkerControlError> {
        match self
            .event_reader
            .next_event()
            .await
            .map_err(WorkerControlError::from)?
        {
            Some(worker_event) => Ok(Some(worker_event)),
            None => Err(self.worker_process_exit_error().await),
        }
    }

    #[must_use]
    pub fn process_id(&self) -> Option<u32> {
        self.worker_process.id()
    }

    /// Closes worker stdin and reaps the process, escalating when it does not exit.
    pub async fn close(&mut self) -> Result<WorkerTerminationOutcome, WorkerControlError> {
        let Some(command_writer) = self.command_writer.take() else {
            return self.force_terminate().await;
        };
        if let Err(close_error) = command_writer.close().await {
            return self.force_after_error(close_error.into()).await;
        }
        match wait_for_worker(&mut self.worker_process, self.shutdown_timeout).await {
            Ok(worker_exit_status) => Ok(WorkerTerminationOutcome::Graceful {
                process_exit_successful: worker_exit_status.success(),
            }),
            Err(shutdown_error) => self.force_after_error(shutdown_error).await,
        }
    }

    pub async fn force_terminate(
        &mut self,
    ) -> Result<WorkerTerminationOutcome, WorkerControlError> {
        let worker_exit_status =
            force_terminate_and_reap(&mut self.worker_process, self.shutdown_timeout).await?;
        Ok(WorkerTerminationOutcome::Forced {
            process_exit_successful: worker_exit_status.success(),
        })
    }

    async fn force_after_error(
        &mut self,
        operation_error: WorkerControlError,
    ) -> Result<WorkerTerminationOutcome, WorkerControlError> {
        match force_terminate_and_reap(&mut self.worker_process, self.shutdown_timeout).await {
            Ok(worker_exit_status) => Ok(WorkerTerminationOutcome::Forced {
                process_exit_successful: worker_exit_status.success(),
            }),
            Err(cleanup_error) => Err(WorkerControlError::OperationAndCleanupFailed {
                operation: Box::new(operation_error),
                cleanup: Box::new(cleanup_error),
            }),
        }
    }

    async fn send_command_with_timeout(
        &mut self,
        worker_command: &WorkerCommand,
    ) -> Result<(), WorkerControlError> {
        let Some(command_writer) = self.command_writer.as_mut() else {
            return Err(WorkerControlError::CommandWriterClosed);
        };
        timeout(
            self.command_write_timeout,
            command_writer.send_command(worker_command),
        )
        .await
        .map_err(|_| WorkerControlError::CommandWriteTimeout {
            command_timeout_millis: self.command_write_timeout.as_millis(),
        })??;
        Ok(())
    }

    /// Collects bounded child diagnostics after stdout can no longer carry IPC.
    async fn worker_process_exit_error(&mut self) -> WorkerControlError {
        // Stdout and stderr are independent pipes. Give the bounded stderr
        // drain a short failure-only interval to consume bytes written beside
        // the final stdout close before snapshotting the diagnostic tail.
        if let Some(mut stderr_drain_task) = self.stderr_drain_task.take() {
            match timeout(
                WORKER_EXIT_DIAGNOSTIC_SETTLE_TIMEOUT,
                &mut stderr_drain_task,
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(stderr_join_error)) => {
                    self.worker_stderr_tail
                        .append(
                            format!("worker stderr drain task failed: {stderr_join_error}")
                                .as_bytes(),
                        )
                        .await;
                }
                Err(_) => {
                    self.stderr_drain_task = Some(stderr_drain_task);
                }
            }
        }
        let process_exit_status = match timeout(
            WORKER_NATURAL_EXIT_GRACE_TIMEOUT,
            self.worker_process.wait(),
        )
        .await
        {
            Ok(Ok(worker_exit_status)) => worker_exit_status_description(worker_exit_status),
            Ok(Err(wait_error)) => {
                format!("failed to inspect process exit status: {wait_error}")
            }
            Err(_) => match self.worker_process.try_wait() {
                Ok(Some(worker_exit_status)) => worker_exit_status_description(worker_exit_status),
                Ok(None) => "process still running after closing stdout".to_owned(),
                Err(wait_error) => {
                    format!("failed to inspect process exit status: {wait_error}")
                }
            },
        };
        WorkerControlError::WorkerProcessExited {
            process_exit_status,
            worker_lifetime_millis: self.worker_started_at.elapsed().as_millis(),
            stderr_tail: self.worker_stderr_tail.diagnostic_snapshot().await,
        }
    }
}

fn spawn_worker_stderr_drain(
    mut worker_stderr: ChildStderr,
    worker_stderr_tail: WorkerStderrTail,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut stderr_read_bytes = [0_u8; WORKER_STDERR_READ_BYTES];
        loop {
            match worker_stderr.read(&mut stderr_read_bytes).await {
                Ok(0) => return,
                Ok(read_byte_count) => {
                    worker_stderr_tail
                        .append(&stderr_read_bytes[..read_byte_count])
                        .await;
                }
                Err(drain_error) => {
                    worker_stderr_tail
                        .append(format!("worker stderr drain failed: {drain_error}").as_bytes())
                        .await;
                    return;
                }
            }
        }
    })
}

fn worker_exit_status_description(worker_exit_status: ExitStatus) -> String {
    match worker_exit_status.code() {
        Some(exit_code) => format!("exit code {exit_code}"),
        None => "terminated by signal".to_owned(),
    }
}

async fn wait_for_worker(
    worker_process: &mut Child,
    shutdown_timeout: Duration,
) -> Result<ExitStatus, WorkerControlError> {
    timeout(shutdown_timeout, worker_process.wait())
        .await
        .map_err(|_| WorkerControlError::ShutdownTimeout {
            shutdown_timeout_millis: shutdown_timeout.as_millis(),
        })?
        .map_err(WorkerControlError::WaitForWorker)
}

async fn cleanup_startup_failure(
    startup_error: WorkerControlError,
    worker_process: &mut Child,
    shutdown_timeout: Duration,
) -> WorkerControlError {
    match force_terminate_and_reap(worker_process, shutdown_timeout).await {
        Ok(_) => startup_error,
        Err(cleanup_error) => WorkerControlError::OperationAndCleanupFailed {
            operation: Box::new(startup_error),
            cleanup: Box::new(cleanup_error),
        },
    }
}

async fn force_terminate_and_reap(
    worker_process: &mut Child,
    shutdown_timeout: Duration,
) -> Result<ExitStatus, WorkerControlError> {
    if let Some(worker_exit_status) = worker_process
        .try_wait()
        .map_err(WorkerControlError::WaitForWorker)?
    {
        return Ok(worker_exit_status);
    }
    match worker_process.start_kill() {
        Ok(()) => {}
        Err(kill_error) if kill_error.kind() == std::io::ErrorKind::NotFound => {}
        Err(kill_error) => return Err(WorkerControlError::TerminateWorker(kill_error)),
    }
    wait_for_worker(worker_process, shutdown_timeout).await
}
