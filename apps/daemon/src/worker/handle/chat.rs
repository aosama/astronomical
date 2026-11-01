//! Connects the public chat-generation executor contract to `WorkerHandle`.
//!
//! Admission remains shared with image generation; this module owns only the
//! chat-facing trait boundary and health projection.

use astronomical_ipc_protocol::ChatGenerationCommand;
use tokio::sync::oneshot;

use crate::serving::ChatGenerationStartFuture;
use crate::{ChatGenerationExecutor, WorkerHandle, WorkerHealthSnapshot, WorkerHealthStatus};

impl ChatGenerationExecutor for WorkerHandle {
    fn start_chat_generation(
        &self,
        chat_generation_command: ChatGenerationCommand,
    ) -> ChatGenerationStartFuture<'_> {
        self.start_chat_generation_with_queue_admission(chat_generation_command, None)
    }

    fn start_chat_generation_with_admission_signal(
        &self,
        chat_generation_command: ChatGenerationCommand,
        admission_sender: oneshot::Sender<()>,
    ) -> ChatGenerationStartFuture<'_> {
        self.start_chat_generation_with_queue_admission(
            chat_generation_command,
            Some(admission_sender),
        )
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        match self.health_snapshot.read() {
            Ok(health_snapshot) => {
                tracing::info!(
                    snapshot_ptr = std::sync::Arc::as_ptr(&self.health_snapshot) as usize,
                    ready_model_id = ?health_snapshot.ready_model_id,
                    observatory_has_latest = health_snapshot
                        .expert_streaming_observatory
                        .latest_stats
                        .is_some(),
                    observatory_transition_count = health_snapshot
                        .expert_streaming_observatory
                        .mode_transitions
                        .len(),
                    "executor health snapshot read"
                );
                health_snapshot.clone()
            }
            Err(_) => WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        }
    }
}
