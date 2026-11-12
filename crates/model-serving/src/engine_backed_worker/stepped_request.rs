//! The requests a worker serves across many loop turns, one step per turn:
//! a chat generation, an image, or speech. Images and speech share how a
//! command that arrives mid-request is answered, so that lives here once.

use astronomical_ipc_protocol::{
    ChatGenerationFailureReason, EmbeddingsFailureReason, ImageGenerationFailureReason,
    ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use tokio::io::{AsyncRead, AsyncWrite};

use super::EngineBackedWorker;
use super::speech::ActiveSpeechGeneration;
use super::support::{ActiveWorkerRequest, ModelFactory, WorkerRuntimeError};
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

/// What a command that arrived during an image or speech request asks of it.
pub(crate) enum CommandDuringRequest {
    /// The command was answered; the request goes on.
    Continue,
    /// The caller cancelled this request.
    Cancel,
}

type ProcessorActiveRequest<Processor> = ActiveWorkerRequest<
    <Processor as ModelGenerationProcessor>::RequestOutput,
    <Processor as ModelGenerationProcessor>::InferenceRequest,
>;

impl<Processor, Engine, Factory, ImageEngine, EmbeddingsEngine, DecisionsEngine, TextToSpeechEngine>
    EngineBackedWorker<
        Processor,
        Engine,
        Factory,
        ImageEngine,
        EmbeddingsEngine,
        DecisionsEngine,
        TextToSpeechEngine,
    >
where
    Processor: ModelGenerationProcessor + Send + 'static,
    Engine: InferenceEngine<Request = Processor::InferenceRequest> + Send + 'static,
    Factory: ModelFactory<
            Processor,
            Engine,
            ImageEngine,
            EmbeddingsEngine,
            DecisionsEngine,
            TextToSpeechEngine,
        > + Send
        + 'static,
    ImageEngine: ImageGenerationEngine,
    EmbeddingsEngine: crate::EmbeddingEngine,
    DecisionsEngine: crate::DecisionEngine,
    TextToSpeechEngine: crate::SpeechEngine,
{
    /// Runs the active request to its next yield.
    pub(crate) async fn serve_active_request<ReadTransport, WriteTransport>(
        &mut self,
        active_request: ProcessorActiveRequest<Processor>,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ProcessorActiveRequest<Processor>>, WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        match active_request {
            ActiveWorkerRequest::Autoregressive(current_generation) => {
                self.serve_active_autoregressive_request(
                    current_generation,
                    command_reader,
                    event_writer,
                )
                .await
            }
            ActiveWorkerRequest::Image(current_generation) => {
                self.serve_active_image_request(current_generation, command_reader, event_writer)
                    .await
            }
            ActiveWorkerRequest::Speech(active_speech) => {
                self.serve_active_speech_request(active_speech, command_reader, event_writer)
                    .await
            }
        }
    }

    /// Starts an image or a speech request from an idle worker.
    pub(crate) async fn start_stepped_request<WriteTransport>(
        &mut self,
        stepped_command: WorkerCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ProcessorActiveRequest<Processor>>, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match stepped_command {
            WorkerCommand::GenerateImage(generation_command) => Ok(self
                .start_image_generation(generation_command, event_writer)
                .await?
                .map(ActiveWorkerRequest::Image)),
            WorkerCommand::GenerateSpeech(speech_command) => Ok(self
                .start_speech(speech_command, event_writer)
                .await?
                .map(ActiveWorkerRequest::Speech)),
            // The idle dispatch routes only those two commands here; anything
            // else would vanish without an answer, so it stops the worker.
            _ => Err(WorkerRuntimeError::InferenceEngineGenerationFailed {
                reason: "a command that is neither an image nor a speech request was routed as one"
                    .to_owned(),
            }),
        }
    }

    /// Speaks one segment per otherwise idle loop turn; a command that
    /// arrives first is answered without disturbing the request, unless it
    /// cancels it.
    async fn serve_active_speech_request<ReadTransport, WriteTransport>(
        &mut self,
        active_speech: ActiveSpeechGeneration,
        command_reader: &mut ProtocolReader<ReadTransport>,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<Option<ProcessorActiveRequest<Processor>>, WorkerRuntimeError>
    where
        ReadTransport: AsyncRead + Unpin,
        WriteTransport: AsyncWrite + Unpin,
    {
        tokio::select! {
            biased;
            next_command = command_reader.next_command() => {
                let command_during_request = match next_command? {
                    Some(worker_command) => {
                        self.answer_command_during_request(worker_command, active_speech.request_id, event_writer).await?
                    }
                    // The daemon closed the channel; nobody is left to hear the audio.
                    None => CommandDuringRequest::Cancel,
                };
                match command_during_request {
                    CommandDuringRequest::Continue => Ok(Some(ActiveWorkerRequest::Speech(active_speech))),
                    CommandDuringRequest::Cancel => {
                        self.cancel_speech(active_speech, event_writer).await?;
                        Ok(None)
                    }
                }
            }
            () = tokio::task::yield_now() => {
                Ok(self
                    .advance_speech(active_speech, event_writer)
                    .await?
                    .map(ActiveWorkerRequest::Speech))
            }
        }
    }

    /// Answers a command that arrived while an image or speech request runs:
    /// another request is refused as busy, a memory limit cannot move, and
    /// a cancel for this request is handed back to the caller.
    pub(crate) async fn answer_command_during_request<WriteTransport>(
        &mut self,
        worker_command: WorkerCommand,
        active_request_id: RequestId,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<CommandDuringRequest, WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match worker_command {
            WorkerCommand::Cancel { request_id } if request_id == active_request_id => {
                return Ok(CommandDuringRequest::Cancel);
            }
            WorkerCommand::Generate(generation_command)
            | WorkerCommand::CountChatTokens(generation_command) => {
                event_writer
                    .send_event(&WorkerEvent::Failed {
                        request_id: generation_command.request_id,
                        reason: ChatGenerationFailureReason::EngineBusy,
                    })
                    .await?;
            }
            WorkerCommand::UpdateMlxMemoryLimit {
                effective_mlx_memory_ceiling_bytes,
                ..
            } => {
                self.reject_memory_limit_during_request(
                    effective_mlx_memory_ceiling_bytes,
                    event_writer,
                )
                .await?;
            }
            other_command => {
                self.refuse_request_while_busy(other_command, event_writer)
                    .await?
            }
        }
        Ok(CommandDuringRequest::Continue)
    }

    /// Refuses an image, embeddings, decisions or speech request because
    /// another request is running. Any other command needs no answer.
    pub(crate) async fn refuse_request_while_busy<WriteTransport>(
        &mut self,
        busy_command: WorkerCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        match busy_command {
            WorkerCommand::GenerateImage(generation_command) => {
                self.emit_image_failure_and_finalization(
                    generation_command.request_id,
                    ImageGenerationFailureReason::EngineBusy,
                    0,
                    event_writer,
                )
                .await
            }
            WorkerCommand::GenerateEmbeddings(embeddings_command) => {
                refuse_embeddings_while_busy(embeddings_command.request_id, event_writer).await
            }
            WorkerCommand::GenerateDecisions(decisions_command) => {
                self.refuse_decisions_while_busy(decisions_command.request_id, event_writer)
                    .await
            }
            WorkerCommand::GenerateSpeech(speech_command) => {
                self.refuse_speech_while_busy(speech_command.request_id, event_writer)
                    .await
            }
            _ => Ok(()),
        }
    }

    async fn reject_memory_limit_during_request<WriteTransport>(
        &self,
        requested_mlx_memory_ceiling_bytes: u64,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        event_writer
            .send_event(&WorkerEvent::MlxMemoryLimitRejected {
                requested_mlx_memory_ceiling_bytes,
                minimum_mlx_memory_ceiling_bytes: self.minimum_mlx_memory_ceiling_bytes,
                machine_mlx_memory_ceiling_bytes: self.machine_mlx_memory_ceiling_bytes,
                reason: "memory limits cannot change during generation".to_owned(),
            })
            .await?;
        Ok(())
    }
}

async fn refuse_embeddings_while_busy<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), WorkerRuntimeError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::EmbeddingsFailed {
            request_id,
            reason: EmbeddingsFailureReason::EngineBusy,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::EmbeddingsFinalized {
            request_id,
            elapsed_millis: 0,
            mlx_memory_snapshot: None,
        })
        .await?;
    Ok(())
}
