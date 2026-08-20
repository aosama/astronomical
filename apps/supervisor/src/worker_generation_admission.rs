//! Owns the one bounded FIFO admission path shared by chat and image requests.
//!
//! Both modalities reserve the same queue and active permits before dispatch,
//! preserving serial execution, cancellation-by-future-drop, and capacity limits.

use std::{future::Future, pin::Pin, sync::Arc};

use astronomical_ipc_protocol::{ChatGenerationCommand, ImageGenerationCommand};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::worker_loop_types::WorkerLoopCommand;
use crate::{
    ChatGenerationStreamEvent, GenerationStartError, ImageGenerationExecutionError,
    ImageGenerationOutput, WorkerHandle,
};

// One slot for every possible 128-token prefill progress boundary plus the
// initial zero-progress event under Qwen3.5-MoE's 262,144-token context limit.
const MAXIMUM_PREFILL_PROGRESS_EVENT_CAPACITY: usize = 2_049;

impl WorkerHandle {
    pub(super) fn start_chat_generation_with_queue_admission(
        &self,
        chat_generation_command: ChatGenerationCommand,
        admission_sender: Option<oneshot::Sender<()>>,
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
            let generation_queue_permit = Arc::clone(&self.generation_queue_permits)
                .try_acquire_owned()
                .map_err(|_| GenerationStartError::CapacityUnavailable)?;
            if let Some(admission_sender) = admission_sender {
                let _admission_signal_result = admission_sender.send(());
            }
            let active_generation_permit = Arc::clone(&self.active_generation_permits)
                .acquire_owned()
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            drop(generation_queue_permit);

            let maximum_output_tokens = chat_generation_command.settings.max_output_tokens;
            let stream_event_capacity = usize::from(maximum_output_tokens)
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

    pub(super) fn start_image_generation_with_queue_admission(
        &self,
        image_generation_command: ImageGenerationCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<
                            Result<ImageGenerationOutput, ImageGenerationExecutionError>,
                        >,
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
            let generation_queue_permit = Arc::clone(&self.generation_queue_permits)
                .try_acquire_owned()
                .map_err(|_| GenerationStartError::CapacityUnavailable)?;
            let admitted_at = Instant::now();
            if let Some(admission_sender) = admission_sender {
                let _admission_signal_result = admission_sender.send(());
            }
            let active_generation_permit = Arc::clone(&self.active_generation_permits)
                .acquire_owned()
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            let queue_wait_elapsed = admitted_at.elapsed();
            drop(generation_queue_permit);

            let (image_result_sender, image_result_receiver) = mpsc::channel(1);
            let (start_sender, start_receiver) = oneshot::channel();
            command_sender
                .send(WorkerLoopCommand::GenerateImage {
                    active_generation_permit,
                    generation_command: image_generation_command,
                    start_sender,
                    image_result_sender,
                    admitted_at,
                    queue_wait_elapsed,
                })
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            start_receiver
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)??;

            Ok(image_result_receiver)
        })
    }
}
