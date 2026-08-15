use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatGenerationOutput,
};
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};

use crate::{WorkerControlError, WorkerHealthSnapshot};

/// Ordered application event produced by one chat request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatGenerationStreamEvent {
    ReasoningFragment(String),
    TextFragment(String),
    ToolCall {
        tool_call_index: u16,
        function_name: String,
        arguments_json: String,
    },
    PrefillProgress {
        processed_tokens: u32,
        total_tokens: u32,
        elapsed_millis: u64,
        forward_prefill_chunk_elapsed_millis: Option<u64>,
        completed_prefill_chunk_tokens: Option<u32>,
        mlx_active_memory_bytes: Option<u64>,
        mlx_allocator_cache_memory_bytes: Option<u64>,
        mlx_peak_memory_bytes: Option<u64>,
    },
    Completed {
        prompt_token_count: u32,
        generated_token_count: u16,
        reasoning_token_count: u16,
        cached_token_count: u32,
        reason: ChatGenerationCompletionReason,
    },
    Failed {
        reason: ChatGenerationFailureReason,
    },
    Error(ChatGenerationStreamErrorCode),
}

impl ChatGenerationStreamEvent {
    pub(crate) fn from_worker_output(worker_output: ChatGenerationOutput) -> Self {
        match worker_output {
            ChatGenerationOutput::Text { text } => Self::TextFragment(text),
            ChatGenerationOutput::Reasoning { text } => Self::ReasoningFragment(text),
            ChatGenerationOutput::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            } => Self::ToolCall {
                tool_call_index,
                function_name,
                arguments_json,
            },
        }
    }
}

/// Supervisor-side reason a chat stream cannot continue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatGenerationStreamErrorCode {
    WorkerUnavailable,
}

/// Failure before a chat stream starts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationStartError {
    CapacityUnavailable,
    ModelLoadFailed {
        model_load_failure_reason: String,
    },
    RequestTooLarge {
        actual_ipc_message_bytes: usize,
        maximum_ipc_message_bytes: usize,
    },
    WorkerUnavailable,
}

/// Starts a bounded chat stream through the one local worker.
pub trait ChatGenerationExecutor: Send + Sync + 'static {
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
    >;

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot;
}

pub(crate) fn try_send_stream_event(
    stream_event_sender: &mpsc::Sender<ChatGenerationStreamEvent>,
    stream_event: ChatGenerationStreamEvent,
) -> Result<(), WorkerControlError> {
    match stream_event_sender.try_send(stream_event) {
        Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => Ok(()),
        Err(mpsc::error::TrySendError::Full(_)) => Err(WorkerControlError::StreamBackpressure),
    }
}

pub(crate) async fn wait_for_stream_disconnect(
    stream_event_sender: Option<mpsc::Sender<ChatGenerationStreamEvent>>,
) {
    let Some(stream_event_sender) = stream_event_sender else {
        std::future::pending::<()>().await;
        return;
    };
    stream_event_sender.closed().await;
}

pub(crate) async fn wait_for_deadline(deadline: Option<Instant>) {
    let Some(deadline) = deadline else {
        std::future::pending::<()>().await;
        return;
    };
    sleep_until(deadline).await;
}
