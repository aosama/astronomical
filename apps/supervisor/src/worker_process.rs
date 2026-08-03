use std::{
    path::Path,
    process::{ExitStatus, Stdio},
    time::Duration,
};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
    WorkerStartupConfiguration,
};
use tokio::{
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command},
    task::JoinHandle,
    time::timeout,
};

use crate::WorkerControlError;

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
    shutdown_timeout: Duration,
    _stderr_drain_task: JoinHandle<()>,
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
        let mut command = Command::new(worker_executable_path.as_ref());
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut worker_process = command.spawn().map_err(WorkerControlError::StartWorker)?;
        let stderr_drain_task = match worker_process.stderr.take() {
            Some(worker_stderr) => spawn_worker_stderr_drain(worker_stderr),
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
            shutdown_timeout,
            _stderr_drain_task: stderr_drain_task,
            worker_process,
        })
    }

    pub async fn start_generation(
        &mut self,
        generation_command: ChatGenerationCommand,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::Generate(generation_command))
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
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes,
        })
        .await
    }

    /// Sends a SwapModel command to the worker, instructing it to unload the current
    /// model and load a new one from the given directory.
    pub async fn swap_model(
        &mut self,
        model_directory: String,
        max_output_tokens: u32,
    ) -> Result<(), WorkerControlError> {
        self.send_command_with_timeout(&WorkerCommand::SwapModel {
            model_directory,
            max_output_tokens,
        })
        .await
    }

    pub async fn next_event(&mut self) -> Result<Option<WorkerEvent>, WorkerControlError> {
        self.event_reader.next_event().await.map_err(Into::into)
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
}

fn spawn_worker_stderr_drain(mut worker_stderr: ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut parent_stderr = tokio::io::stderr();
        if let Err(drain_error) = tokio::io::copy(&mut worker_stderr, &mut parent_stderr).await {
            eprintln!("worker stderr drain failed: {drain_error}");
        }
    })
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
    match worker_process.start_kill() {
        Ok(()) => {}
        Err(kill_error) if kill_error.kind() == std::io::ErrorKind::NotFound => {}
        Err(kill_error) => return Err(WorkerControlError::TerminateWorker(kill_error)),
    }
    wait_for_worker(worker_process, shutdown_timeout).await
}
