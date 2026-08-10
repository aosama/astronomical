use std::{
    collections::HashMap,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, RwLock},
    time::Duration,
};

use crate::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    GenerationStartError, WorkerControlError, WorkerHealthSnapshot, WorkerHealthStatus,
    WorkerProcess, WorkerTerminationOutcome,
};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, WorkerRuntimeFeatureConfiguration, WorkerStartupConfiguration,
};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio::time::sleep;

use crate::worker::run_worker;
use crate::worker_loop_types::WorkerLoopCommand;
use crate::worker_memory_limit::MlxMemoryLimitUpdateOutcome;

const WORKER_COMMAND_CAPACITY: usize = 8;
const DEFAULT_WORKER_CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT: Duration = Duration::from_secs(60);
// One slot for every possible 128-token prefill progress boundary plus the
// initial zero-progress event under Qwen3.5-MoE's 262,144-token context limit.
const MAXIMUM_PREFILL_PROGRESS_EVENT_CAPACITY: usize = 2_049;

/// How many requests can wait in the queue when one request is already active.
/// A second request waits in the queue; the (QUEUE_DEPTH + 1)th request is
/// rejected immediately with CapacityUnavailable.
const GENERATION_QUEUE_DEPTH: usize = 8;

/// Exposes the generation queue depth constant for tests and configuration.
pub struct GenerationQueueDepth;

impl GenerationQueueDepth {
    /// Returns the maximum number of requests that can wait in the queue
    /// when one request is already active.
    #[must_use]
    pub const fn value() -> usize {
        GENERATION_QUEUE_DEPTH
    }
}

/// Cloneable HTTP-side handle for the one local worker process.
///
/// Requests are processed serially (one at a time). When a request is active,
/// up to `GENERATION_QUEUE_DEPTH` additional requests wait in a FIFO queue.
/// Requests beyond the queue depth are rejected immediately with
/// `CapacityUnavailable`.
#[derive(Clone)]
pub struct WorkerHandle {
    /// Permit for the active generation slot. Only one generation runs at a time.
    /// Acquired (awaited) by the next queued request when the current generation
    /// completes and releases its permit.
    active_generation_permits: Arc<Semaphore>,
    /// Permits for the FIFO queue. Each queued request reserves one permit;
    /// when the active slot becomes available, the permit is promoted to an
    /// active-generation permit and the queue permit is released.
    generation_queue_permits: Arc<Semaphore>,
    command_sender: Option<mpsc::Sender<WorkerLoopCommand>>,
    health_snapshot: Arc<RwLock<WorkerHealthSnapshot>>,
}

impl WorkerHandle {
    /// Creates a handle that reports an unavailable worker without starting a process.
    #[must_use]
    pub fn unavailable() -> Self {
        Self {
            active_generation_permits: Arc::new(Semaphore::new(1)),
            generation_queue_permits: Arc::new(Semaphore::new(GENERATION_QUEUE_DEPTH)),
            command_sender: None,
            health_snapshot: Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
                WorkerHealthStatus::Unavailable,
            ))),
        }
    }

    /// Launches one idle worker with explicit load timeout and model lookup for hot-swap.
    pub async fn launch(
        worker_executable_path: impl AsRef<Path>,
        worker_model_load_timeout: Duration,
        performance_log: GenerationPerformanceLog,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_optional_startup_configuration(
            worker_executable_path,
            worker_model_load_timeout,
            DEFAULT_WORKER_CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT,
            performance_log,
            model_directories,
            max_output_tokens,
            None,
        )
        .await
    }

    /// Launches a production worker with supervisor-resolved bootstrap settings.
    pub async fn launch_with_startup_configuration(
        worker_executable_path: impl AsRef<Path>,
        worker_model_load_timeout: Duration,
        performance_log: GenerationPerformanceLog,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
        worker_startup_configuration: WorkerStartupConfiguration,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_optional_startup_configuration(
            worker_executable_path,
            worker_model_load_timeout,
            DEFAULT_WORKER_CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT,
            performance_log,
            model_directories,
            max_output_tokens,
            Some(worker_startup_configuration),
        )
        .await
    }

    /// Launches one idle worker with an explicit cancellation acknowledgement timeout.
    pub async fn launch_with_cancellation_acknowledgement_timeout(
        worker_executable_path: impl AsRef<Path>,
        worker_model_load_timeout: Duration,
        worker_cancellation_acknowledgement_timeout: Duration,
        performance_log: GenerationPerformanceLog,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
    ) -> Result<Self, WorkerControlError> {
        Self::launch_with_optional_startup_configuration(
            worker_executable_path,
            worker_model_load_timeout,
            worker_cancellation_acknowledgement_timeout,
            performance_log,
            model_directories,
            max_output_tokens,
            None,
        )
        .await
    }

    async fn launch_with_optional_startup_configuration(
        worker_executable_path: impl AsRef<Path>,
        worker_model_load_timeout: Duration,
        worker_cancellation_acknowledgement_timeout: Duration,
        performance_log: GenerationPerformanceLog,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
        worker_startup_configuration: Option<WorkerStartupConfiguration>,
    ) -> Result<Self, WorkerControlError> {
        let worker_process = match worker_startup_configuration {
            Some(worker_startup_configuration) => {
                WorkerProcess::launch_with_startup_configuration(
                    worker_executable_path.as_ref(),
                    worker_startup_configuration,
                )
                .await?
            }
            None => WorkerProcess::launch(worker_executable_path.as_ref()).await?,
        };
        let (command_sender, command_receiver) = mpsc::channel(WORKER_COMMAND_CAPACITY);
        let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
            WorkerHealthStatus::Loading,
        )));

        tokio::spawn(run_worker(
            worker_process,
            command_receiver,
            Arc::clone(&health_snapshot),
            worker_model_load_timeout,
            worker_cancellation_acknowledgement_timeout,
            performance_log,
            model_directories,
            max_output_tokens,
        ));

        Ok(Self {
            active_generation_permits: Arc::new(Semaphore::new(1)),
            generation_queue_permits: Arc::new(Semaphore::new(GENERATION_QUEUE_DEPTH)),
            command_sender: Some(command_sender),
            health_snapshot,
        })
    }

    /// Shuts down and reaps the owned inference worker process.
    pub async fn shutdown(self) -> Result<WorkerTerminationOutcome, WorkerControlError> {
        let Some(command_sender) = self.command_sender else {
            return Ok(WorkerTerminationOutcome::Graceful {
                process_exit_successful: true,
            });
        };
        let (shutdown_sender, shutdown_receiver) = oneshot::channel();
        command_sender
            .send(WorkerLoopCommand::Shutdown { shutdown_sender })
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?;

        shutdown_receiver
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?
    }

    /// Returns whether no generation is active or waiting in the FIFO queue.
    #[must_use]
    pub fn is_generation_idle_for_control_action(&self) -> bool {
        self.active_generation_permits.available_permits() == 1
            && self.generation_queue_permits.available_permits() == GENERATION_QUEUE_DEPTH
    }

    /// Replaces the owned worker process while keeping the REST application alive.
    pub async fn restart_worker(
        &self,
        worker_executable_path: PathBuf,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
    ) -> Result<(), WorkerControlError> {
        self.restart_worker_with_optional_startup_configuration(
            worker_executable_path,
            model_directories,
            max_output_tokens,
            None,
        )
        .await
    }

    /// Replaces the production worker using supervisor-resolved bootstrap settings.
    pub async fn restart_worker_with_startup_configuration(
        &self,
        worker_executable_path: PathBuf,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
        worker_startup_configuration: WorkerStartupConfiguration,
    ) -> Result<(), WorkerControlError> {
        self.restart_worker_with_optional_startup_configuration(
            worker_executable_path,
            model_directories,
            max_output_tokens,
            Some(worker_startup_configuration),
        )
        .await
    }

    async fn restart_worker_with_optional_startup_configuration(
        &self,
        worker_executable_path: PathBuf,
        model_directories: Arc<HashMap<String, PathBuf>>,
        max_output_tokens: u32,
        worker_startup_configuration: Option<WorkerStartupConfiguration>,
    ) -> Result<(), WorkerControlError> {
        if !self.is_generation_idle_for_control_action() {
            return Err(WorkerControlError::GenerationBusy);
        }
        let command_sender = self
            .command_sender
            .as_ref()
            .ok_or(WorkerControlError::MissingActiveWorker)?;
        let (restart_sender, restart_receiver) = oneshot::channel();
        command_sender
            .send(WorkerLoopCommand::RestartWorker {
                worker_executable_path,
                model_directories,
                max_output_tokens,
                worker_startup_configuration,
                restart_sender,
            })
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?;
        restart_receiver
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?
    }

    /// Waits for the replacement worker to acknowledge its runtime feature policy.
    pub async fn wait_for_worker_runtime_feature_configuration(
        &self,
        acknowledgement_timeout: Duration,
    ) -> Result<WorkerRuntimeFeatureConfiguration, WorkerControlError> {
        tokio::time::timeout(acknowledgement_timeout, async {
            loop {
                let worker_health_snapshot = self.worker_health_snapshot();
                // Restart publishes Loading before the new process can send any event, so an
                // old worker acknowledgement cannot satisfy this wait. Ready plus the explicit
                // configuration event is the one complete replacement handshake.
                if worker_health_snapshot.status == WorkerHealthStatus::Unavailable {
                    return Err(WorkerControlError::MissingActiveWorker);
                }
                if worker_health_snapshot.status == WorkerHealthStatus::Ready
                    && let Some(worker_runtime_feature_configuration) =
                        worker_health_snapshot.worker_runtime_feature_configuration
                {
                    return Ok(worker_runtime_feature_configuration);
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .map_err(|_| WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: acknowledgement_timeout.as_millis(),
        })?
    }

    /// Applies an idle MLX ceiling immediately or queues it behind one active request.
    pub async fn update_mlx_memory_limit(
        &self,
        effective_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
        let command_sender = self
            .command_sender
            .as_ref()
            .ok_or(WorkerControlError::MissingActiveWorker)?;
        let (update_sender, update_receiver) = oneshot::channel();
        command_sender
            .send(WorkerLoopCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
                update_sender,
            })
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?;
        update_receiver
            .await
            .map_err(|_| WorkerControlError::MissingActiveWorker)?
    }
}

impl ChatGenerationExecutor for WorkerHandle {
    fn start_chat_generation(
        &self,
        chat_generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let command_sender = self
                .command_sender
                .as_ref()
                .ok_or(GenerationStartError::WorkerUnavailable)?;

            // Reserve a slot in the FIFO queue. If the queue is full, reject
            // immediately with CapacityUnavailable rather than blocking.
            let generation_queue_permit = Arc::clone(&self.generation_queue_permits)
                .try_acquire_owned()
                .map_err(|_| GenerationStartError::CapacityUnavailable)?;

            // Wait for the active-generation slot to become free. This serializes
            // requests: only one runs at a time, and queued requests proceed in
            // FIFO order as each previous generation completes.
            let active_generation_permit = Arc::clone(&self.active_generation_permits)
                .acquire_owned()
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;

            // Promoted from queue to active: release the queue reservation so
            // another request can enter the queue.
            drop(generation_queue_permit);

            let max_output_tokens = chat_generation_command.settings.max_output_tokens;
            let stream_event_capacity = usize::from(max_output_tokens)
                .saturating_add(MAXIMUM_PREFILL_PROGRESS_EVENT_CAPACITY)
                .max(1);
            let (stream_event_sender, stream_event_receiver) = mpsc::channel(stream_event_capacity);
            let (start_sender, start_receiver) = oneshot::channel();

            command_sender
                .send(WorkerLoopCommand::Generate {
                    active_generation_permit,
                    generation_command: chat_generation_command,
                    start_sender,
                    stream_event_sender,
                })
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;

            start_receiver
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)??;

            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        match self.health_snapshot.read() {
            Ok(health_snapshot) => health_snapshot.clone(),
            Err(_) => WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        }
    }
}
