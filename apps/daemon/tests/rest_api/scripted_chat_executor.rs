//! The scripted chat worker every generation-surface journey drives: the
//! OpenAI Responses and the Anthropic Messages endpoints run the same
//! generation path, so they share one stand-in for the worker.

use std::{future::Future, pin::Pin};

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
};
use astronomical_ipc_protocol::{ChatGenerationCommand, ChatModelCapabilities};
use tokio::sync::mpsc;

/// The one model the scripted worker reports as ready.
pub(crate) const SCRIPTED_CHAT_MODEL_ID: &str = "astronomical/responses-endpoint-test-model";

/// A worker stand-in that answers every generation with a fixed event script
/// (or a fixed start refusal) and reports one ready chat model.
pub(crate) struct ScriptedChatExecutor {
    stream_events: Vec<ChatGenerationStreamEvent>,
    start_error: Option<GenerationStartError>,
    expected_model_id: Option<&'static str>,
    is_idle_without_model: bool,
}

impl ScriptedChatExecutor {
    pub(crate) fn new(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            stream_events,
            start_error: None,
            expected_model_id: None,
            is_idle_without_model: false,
        }
    }

    pub(crate) fn with_start_error(start_error: GenerationStartError) -> Self {
        Self {
            stream_events: Vec::new(),
            start_error: Some(start_error),
            expected_model_id: None,
            is_idle_without_model: false,
        }
    }

    pub(crate) fn idle_with_expected_model(expected_model_id: &'static str) -> Self {
        Self {
            stream_events: Vec::new(),
            start_error: Some(GenerationStartError::CapacityUnavailable),
            expected_model_id: Some(expected_model_id),
            is_idle_without_model: true,
        }
    }
}

impl ChatGenerationExecutor for ScriptedChatExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
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
            if let Some(expected_model_id) = self.expected_model_id {
                assert_eq!(generation_command.model, expected_model_id);
            }
            if let Some(start_error) = &self.start_error {
                return Err(start_error.clone());
            }
            let (stream_event_sender, stream_event_receiver) =
                mpsc::channel(self.stream_events.len().max(1));
            for stream_event in &self.stream_events {
                if stream_event_sender
                    .send(stream_event.clone())
                    .await
                    .is_err()
                {
                    return Err(GenerationStartError::WorkerUnavailable);
                }
            }
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        if self.is_idle_without_model {
            return WorkerHealthSnapshot::ready_without_model(0);
        }
        WorkerHealthSnapshot::ready_with_model(
            SCRIPTED_CHAT_MODEL_ID.to_owned(),
            ChatModelCapabilities {
                supports_reasoning: true,
                supports_tool_calls: true,
                has_vision: true,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
        )
    }
}

impl astronomical_daemon::ImageGenerationExecutor for ScriptedChatExecutor {}
