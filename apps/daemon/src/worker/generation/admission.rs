//! Owns the one bounded FIFO admission path shared by chat, image, and embedding requests.
//!
//! All modalities reserve the same queue and active permits before dispatch,
//! preserving serial execution, cancellation-by-future-drop, and capacity limits.

use std::sync::Arc;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, DecisionsCommand, EmbeddingsCommand, ImageGenerationCommand,
    SpeechCommand,
};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::serving::{ChatGenerationStartFuture, GenerationOutcomeStartFuture};
use crate::worker::loop_types::{ChatWork, QueuedChatRequest, WorkerLoopCommand};
use crate::{
    DecisionsExecutionError, DecisionsOutput, EmbeddingsExecutionError, EmbeddingsOutput,
    GenerationStartError, ImageGenerationExecutionError, ImageGenerationOutput,
    SpeechExecutionError, SpeechOutput, WorkerHandle,
};

// One slot for every possible 128-token prefill progress boundary plus the
// initial zero-progress event under Qwen3.5-MoE's 262,144-token context limit.
const MAXIMUM_PREFILL_PROGRESS_EVENT_CAPACITY: usize = 2_049;

impl WorkerHandle {
    pub(in crate::worker) fn start_chat_generation_with_queue_admission(
        &self,
        chat_generation_command: ChatGenerationCommand,
        chat_work: ChatWork,
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
                    queued_chat_request: QueuedChatRequest {
                        generation_command: chat_generation_command,
                        chat_work,
                    },
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

    /// Queues one decisions request behind whatever already waits for the
    /// worker, and resolves once the worker has accepted it.
    pub(in crate::worker) fn start_decisions_generation_with_queue_admission(
        &self,
        decisions_command: DecisionsCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> GenerationOutcomeStartFuture<'_, DecisionsOutput, DecisionsExecutionError> {
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

            let (decisions_result_sender, decisions_result_receiver) = mpsc::channel(1);
            let (start_sender, start_receiver) = oneshot::channel();
            command_sender
                .send(WorkerLoopCommand::GenerateDecisions {
                    active_generation_permit,
                    decisions_command,
                    start_sender,
                    decisions_result_sender,
                })
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            // Two failure layers: a closed channel means the worker loop died;
            // the inner Result is the generation admission outcome.
            let start_acknowledgement_outcome = match start_receiver.await {
                Ok(acknowledgement_outcome) => acknowledgement_outcome,
                Err(_channel_closed) => return Err(GenerationStartError::WorkerUnavailable),
            };
            start_acknowledgement_outcome?;

            Ok(decisions_result_receiver)
        })
    }

    /// Queues one speech request behind the same permits as every other
    /// request, then hands it to the worker loop.
    pub(in crate::worker) fn start_speech_generation_with_queue_admission(
        &self,
        speech_command: SpeechCommand,
        admission_sender: Option<oneshot::Sender<()>>,
    ) -> GenerationOutcomeStartFuture<'_, SpeechOutput, SpeechExecutionError> {
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

            let (speech_result_sender, speech_result_receiver) = mpsc::channel(1);
            let (start_sender, start_receiver) = oneshot::channel();
            command_sender
                .send(WorkerLoopCommand::GenerateSpeech {
                    active_generation_permit,
                    speech_command,
                    start_sender,
                    speech_result_sender,
                })
                .await
                .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            match start_receiver.await {
                Ok(start_acknowledgement_outcome) => start_acknowledgement_outcome?,
                Err(_channel_closed) => return Err(GenerationStartError::WorkerUnavailable),
            }
            Ok(speech_result_receiver)
        })
    }
}
