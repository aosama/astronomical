//! Owns the one bounded FIFO admission path shared by chat, image, and embedding requests.
//!
//! All modalities reserve the same queue and active permits before dispatch,
//! preserving serial execution, cancellation-by-future-drop, and capacity limits.

use std::sync::Arc;

use astronomical_ipc_protocol::{ChatGenerationCommand, EmbeddingsCommand, ImageGenerationCommand};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::serving::{ChatGenerationStartFuture, GenerationOutcomeStartFuture};
use crate::worker::loop_types::WorkerLoopCommand;
use crate::{
    EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, WorkerHandle,
};

// One slot for every possible 128-token prefill progress boundary plus the
// initial zero-progress event under Qwen3.5-MoE's 262,144-token context limit.
const MAXIMUM_PREFILL_PROGRESS_EVENT_CAPACITY: usize = 2_049;

impl WorkerHandle {
    pub(in crate::worker) fn start_chat_generation_with_queue_admission(
        &self,
        chat_generation_command: ChatGenerationCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> ChatGenerationStartFuture<'_> {
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
            // Two failure layers: a closed channel means the worker loop died;
            // the inner Result is the generation admission outcome.
            let start_acknowledgement_join = start_receiver.await;
            let start_acknowledgement_outcome = match start_acknowledgement_join {
                Ok(acknowledgement_outcome) => acknowledgement_outcome,
                Err(_channel_closed) => return Err(GenerationStartError::WorkerUnavailable),
            };
            start_acknowledgement_outcome?;

            Ok(stream_event_receiver)
        })
    }

    pub(in crate::worker) fn start_image_generation_with_queue_admission(
        &self,
        image_generation_command: ImageGenerationCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> GenerationOutcomeStartFuture<'_, ImageGenerationOutput, ImageGenerationExecutionError>
    {
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
            // Two failure layers: a closed channel means the worker loop died;
            // the inner Result is the generation admission outcome.
            let start_acknowledgement_join = start_receiver.await;
            let start_acknowledgement_outcome = match start_acknowledgement_join {
                Ok(acknowledgement_outcome) => acknowledgement_outcome,
                Err(_channel_closed) => return Err(GenerationStartError::WorkerUnavailable),
            };
            start_acknowledgement_outcome?;

            Ok(image_result_receiver)
        })
    }

    pub(in crate::worker) fn start_embeddings_generation_with_queue_admission(
        &self,
        embeddings_command: EmbeddingsCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> GenerationOutcomeStartFuture<'_, EmbeddingsOutput, EmbeddingsExecutionError> {
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

            let (embeddings_result_sender, embeddings_result_receiver) = mpsc::channel(1);
            let (start_sender, start_receiver) = oneshot::channel();
            command_sender
                .send(WorkerLoopCommand::GenerateEmbeddings {
                    active_generation_permit,
                    embeddings_command,
                    start_sender,
                    embeddings_result_sender,
                    admitted_at,
                    queue_wait_elapsed,
                })
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            // Two failure layers: a closed channel means the worker loop died;
            // the inner Result is the generation admission outcome.
            let start_acknowledgement_join = start_receiver.await;
            let start_acknowledgement_outcome = match start_acknowledgement_join {
                Ok(acknowledgement_outcome) => acknowledgement_outcome,
                Err(_channel_closed) => return Err(GenerationStartError::WorkerUnavailable),
            };
            start_acknowledgement_outcome?;

            Ok(embeddings_result_receiver)
        })
    }
}
