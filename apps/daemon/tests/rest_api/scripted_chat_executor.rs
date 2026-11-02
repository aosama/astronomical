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
    /// The script a token count replays; `None` refuses counts as a worker
    /// without a count path would.
    token_count_events: Option<Vec<ChatGenerationStreamEvent>>,
}

impl ScriptedChatExecutor {
    pub(crate) fn new(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            stream_events,
            start_error: None,
            expected_model_id: None,
            is_idle_without_model: false,
            token_count_events: None,
        }
    }

    pub(crate) fn with_start_error(start_error: GenerationStartError) -> Self {
        Self {
            stream_events: Vec::new(),
            start_error: Some(start_error),
            expected_model_id: None,
            is_idle_without_model: false,
            token_count_events: None,
        }
    }

    pub(crate) fn idle_with_expected_model(expected_model_id: &'static str) -> Self {
        Self {
            stream_events: Vec::new(),
            start_error: Some(GenerationStartError::CapacityUnavailable),
            expected_model_id: Some(expected_model_id),
            is_idle_without_model: true,
            token_count_events: None,
        }
    }

    /// Answers token counts with `token_count_events` and refuses every
    /// generation, so a journey proves its request took the count path.
    pub(crate) fn answering_token_counts(
        token_count_events: Vec<ChatGenerationStreamEvent>,
    ) -> Self {
        Self {
            stream_events: Vec::new(),
            start_error: Some(GenerationStartError::WorkerUnavailable),
            expected_model_id: None,
            is_idle_without_model: false,
            token_count_events: Some(token_count_events),
        }
    }
}

async fn replay_stream_events(
    stream_events: &[ChatGenerationStreamEvent],
) -> Result<mpsc::Receiver<ChatGenerationStreamEvent>, GenerationStartError> {
    let (stream_event_sender, stream_event_receiver) = mpsc::channel(stream_events.len().max(1));
    for stream_event in stream_events {
        if stream_event_sender
            .send(stream_event.clone())
            .await
            .is_err()
        {
            return Err(GenerationStartError::WorkerUnavailable);
        }
    }
    Ok(stream_event_receiver)
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
            replay_stream_events(&self.stream_events).await
        })
    }

    fn start_chat_token_count_with_admission_signal(
        &self,
        count_command: ChatGenerationCommand,
        admission_sender: tokio::sync::oneshot::Sender<()>,
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
        let _admission_signal_outcome = admission_sender.send(());
        Box::pin(async move {
            // A count never generates, so it asks for the smallest output budget.
            assert_eq!(count_command.settings.max_output_tokens, 1);
            // Without a count script, a configured start refusal (a full
            // queue, a failed model load) answers counts the way it answers
            // generations, because both share one admission queue.
            match (&self.token_count_events, &self.start_error) {
                (Some(token_count_events), _) => replay_stream_events(token_count_events).await,
                (None, Some(start_error)) => Err(start_error.clone()),
                (None, None) => Err(GenerationStartError::WorkerUnavailable),
            }
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
