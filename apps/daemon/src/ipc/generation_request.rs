//! Routes the daemon IPC requests a worker model answers with generated
//! output (chat, embeddings, images) to their handlers.

use ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DaemonTransportError, ImageGenerateRequest,
    StreamingResponseWriter,
};

use super::service::DaemonIpcGenerationContext;
use crate::performance::attribution::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};

/// The requests a worker model answers with generated output. The dispatch in
/// `service` builds one in its own arms, so both matches stay exhaustive: a
/// new `DaemonRequest` variant fails to compile in the dispatch, and a new
/// kind added here fails to compile in the routing match, instead of a catch-all
/// silently closing the client's stream.
pub(super) enum GenerationRequest {
    Chat(super::chat::DaemonIpcChatRequest),
    Embed {
        model: Option<String>,
        inputs: Vec<String>,
        dimensions: Option<u32>,
    },
    Image(ImageGenerateRequest),
}

impl GenerationRequest {
    /// Groups the four chat fields of a `ChatGenerate` request.
    pub(super) fn chat(
        model: String,
        messages: Vec<ChatMessage>,
        settings: ChatGenerationSettings,
        schema_json: Option<String>,
    ) -> Self {
        Self::Chat(super::chat::DaemonIpcChatRequest {
            model,
            messages,
            settings,
            schema_json,
        })
    }

    /// Carries the fields of an `EmbedGenerate` request.
    pub(super) fn embed(
        model: Option<String>,
        inputs: Vec<String>,
        dimensions: Option<u32>,
    ) -> Self {
        Self::Embed {
            model,
            inputs,
            dimensions,
        }
    }
}

/// Runs a request a worker model answers with generated output: a chat
/// reply, embeddings or an image.
pub(super) async fn handle_generation_request(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    generation_request: GenerationRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    match generation_request {
        GenerationRequest::Chat(chat_request) => {
            stream_chat_generate(
                generation_context,
                daemon_attribution_log,
                chat_request,
                streaming_response_writer,
            )
            .await
        }
        GenerationRequest::Embed {
            model,
            inputs,
            dimensions,
        } => {
            crate::ipc::embeddings::run_embeddings_generation(
                generation_context,
                daemon_attribution_log,
                model,
                inputs,
                dimensions,
                streaming_response_writer,
            )
            .await
        }
        GenerationRequest::Image(image_request) => {
            crate::ipc::images::run_image_generation(
                generation_context,
                image_request,
                streaming_response_writer,
            )
            .await
        }
    }
}

/// Streams one chat generation under its attributed measurement region.
async fn stream_chat_generate(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    chat_request: super::chat::DaemonIpcChatRequest,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcChatGenerate,
            || {
                super::chat::stream_chat_generation(
                    generation_context,
                    chat_request,
                    streaming_response_writer,
                )
            },
            |stream_outcome| {
                if stream_outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                }
            },
        )
        .await
}
