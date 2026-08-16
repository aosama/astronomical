use std::io;

use astronomical_ipc_protocol::ProtocolError;
use thiserror::Error;

/// Errors raised while the supervisor launches, communicates with, or reaps a worker process.
#[derive(Debug, Error)]
pub enum WorkerControlError {
    /// The operation and its best-effort cleanup both failed.
    #[error("worker operation failed: {operation}; cleanup also failed: {cleanup}")]
    OperationAndCleanupFailed {
        /// Original typed operation failure retained as the error source.
        #[source]
        operation: Box<WorkerControlError>,
        /// Typed cleanup failure retained for explicit inspection.
        cleanup: Box<WorkerControlError>,
    },

    /// The application detected an invalid operational event correlation or sequence.
    #[error("worker protocol event violated request expectations: {description}")]
    WorkerProtocolViolation {
        /// Bounded diagnostic describing the violated event expectation.
        description: &'static str,
    },

    /// A bounded IPC command write did not complete before its deadline.
    #[error(
        "worker command write did not complete within the {command_timeout_millis}-millisecond timeout"
    )]
    CommandWriteTimeout {
        /// The maximum permitted command write interval.
        command_timeout_millis: u128,
    },

    /// The worker process no longer has a command writer available for IPC.
    #[error("worker process command writer is no longer available")]
    CommandWriterClosed,

    /// The HTTP stream stopped consuming output before its bounded channel filled.
    #[error("HTTP chat stream stopped consuming worker output")]
    StreamBackpressure,

    /// The worker did not acknowledge cancellation within the configured containment interval.
    #[error(
        "worker cancellation acknowledgement did not arrive within the {cancellation_timeout_millis}-millisecond cancellation timeout"
    )]
    CancellationAckTimeout {
        /// The maximum permitted cancellation acknowledgement interval.
        cancellation_timeout_millis: u128,
    },

    /// The worker remained responsive but did not finish loading the inference engine in time.
    #[error(
        "worker did not finish loading the inference engine within the {model_load_timeout_millis}-millisecond timeout"
    )]
    ModelLoadTimeout {
        /// The maximum permitted model-loading interval.
        model_load_timeout_millis: u128,
    },

    /// The worker did not acknowledge a live memory-limit update in time.
    #[error(
        "worker memory-limit update acknowledgement did not arrive within the {memory_limit_update_timeout_millis}-millisecond timeout"
    )]
    MlxMemoryLimitUpdateTimeout {
        /// Maximum interval permitted for the worker-side adjustment.
        memory_limit_update_timeout_millis: u128,
    },
    /// The worker did not acknowledge a prompt-cache clear in time.
    #[error(
        "worker prompt-cache clear acknowledgement did not arrive within the {cache_clear_timeout_millis}-millisecond timeout"
    )]
    PromptCacheClearTimeout { cache_clear_timeout_millis: u128 },

    /// The supervisor no longer owns a worker process to operate on.
    #[error("supervisor does not currently own an active worker process")]
    MissingActiveWorker,

    /// A control action was rejected because generation work is active or queued.
    #[error("worker control action requires an idle generation queue")]
    GenerationBusy,

    /// Sending a forced termination signal to the worker failed.
    #[error("failed to force-terminate worker process")]
    TerminateWorker(#[source] io::Error),

    /// The worker emitted an unexpected event while cancellation cleanup was waiting for an ack.
    #[error(
        "worker emitted an unexpected event while cancelling request {request_id}: {unexpected_worker_event_summary}"
    )]
    UnexpectedCancellationEvent {
        /// The request being cancelled.
        request_id: u64,
        /// A bounded event-kind and request-correlation summary without model payloads.
        unexpected_worker_event_summary: String,
    },

    /// A non-process fixture closed its IPC output without process diagnostics.
    #[error("worker closed its event stream before the expected event")]
    WorkerEventStreamClosed,

    /// The worker process closed IPC and supplied bounded process diagnostics.
    #[error(
        "worker process exited after closing its event stream ({process_exit_status}) after {worker_lifetime_millis} milliseconds; worker stderr tail: {stderr_tail}"
    )]
    WorkerProcessExited {
        /// Exit code, signal, or a statement that the process had not exited yet.
        process_exit_status: String,
        /// End-to-end lifetime from successful process spawn through IPC closure.
        worker_lifetime_millis: u128,
        /// Newest bounded bytes drained from worker standard error.
        stderr_tail: String,
    },

    /// The child process did not expose a writable stdin pipe.
    #[error("worker process did not expose stdin")]
    MissingStandardInput,

    /// The child process did not expose a readable stdout pipe.
    #[error("worker process did not expose stdout")]
    MissingStandardOutput,

    /// The child process did not expose a readable stderr pipe for diagnostic draining.
    #[error("worker process did not expose stderr")]
    MissingStandardError,

    /// A bounded IPC protocol operation failed.
    #[error("worker protocol operation failed: {0}")]
    Protocol(#[from] ProtocolError),

    /// Starting the worker process failed.
    #[error("failed to start worker process")]
    StartWorker(#[source] io::Error),

    /// The worker did not exit during the configured shutdown grace period.
    #[error(
        "worker did not exit within the {shutdown_timeout_millis}-millisecond shutdown timeout"
    )]
    ShutdownTimeout {
        /// The maximum permitted graceful-shutdown interval.
        shutdown_timeout_millis: u128,
    },

    /// Reaping the worker process failed.
    #[error("failed to wait for worker process exit")]
    WaitForWorker(#[source] io::Error),
}
