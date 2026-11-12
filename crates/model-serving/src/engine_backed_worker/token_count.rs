//! Serves `CountChatTokens`: the prompt token count a chat request would
//! prefill, measured by running only the generation preparation (chat
//! template, tokenizer, and image-token expansion) and never the model, so
//! the count is the exact number a `Generate` of the same request prefills.

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ProtocolWriter, WorkerEvent,
};
use tokio::io::AsyncWrite;

use super::generation_start::protocol_prompt_token_count;
use super::support::{ModelFactory, WorkerRuntimeError};
use crate::inference_engine::contract::PreparedInferenceRequest;
use crate::model_generation_processor::ModelGenerationProcessor;
use crate::{ImageGenerationEngine, InferenceEngine};

use super::EngineBackedWorker;

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
    /// Answers with one `Completed` event carrying the prepared prompt's
    /// token count and zero generated tokens. Preparation failures were
    /// already answered with `Failed` by the shared preparation step.
    pub(super) async fn count_chat_tokens<WriteTransport>(
        &self,
        count_command: ChatGenerationCommand,
        event_writer: &mut ProtocolWriter<WriteTransport>,
    ) -> Result<(), WorkerRuntimeError>
    where
        WriteTransport: AsyncWrite + Unpin,
    {
        let request_id = count_command.request_id;
        let Some(prepared_generation) = self
            .prepare_validated_chat_generation(&count_command, request_id, event_writer)
            .await?
        else {
            return Ok(());
        };
        let prompt_token_count = protocol_prompt_token_count(
            prepared_generation.inference_request.prompt_token_count(),
        )?;
        event_writer
            .send_event(&WorkerEvent::Completed {
                request_id,
                prompt_token_count,
                generated_token_count: 0,
                reasoning_token_count: 0,
                cached_token_count: 0,
                persistent_prompt_cache_diagnostics: None,
                reason: ChatGenerationCompletionReason::EndOfSequence,
            })
            .await?;
        Ok(())
    }
}
