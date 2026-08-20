//! Connects the public chat-generation executor contract to `WorkerHandle`.
//!
//! Admission remains shared with image generation; this module owns only the
//! chat-facing trait boundary and health projection.

use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::ChatGenerationCommand;
use tokio::sync::{mpsc, oneshot};

use crate::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHandle,
    WorkerHealthSnapshot, WorkerHealthStatus,
};

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
        self.start_chat_generation_with_queue_admission(chat_generation_command, None)
    }

    fn start_chat_generation_with_admission_signal(
        &self,
        chat_generation_command: ChatGenerationCommand,
        admission_sender: oneshot::Sender<()>,
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
        self.start_chat_generation_with_queue_admission(
            chat_generation_command,
            Some(admission_sender),
        )
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        match self.health_snapshot.read() {
            Ok(health_snapshot) => health_snapshot.clone(),
            Err(_) => WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        }
    }
}
