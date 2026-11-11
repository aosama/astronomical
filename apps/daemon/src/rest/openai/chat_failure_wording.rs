//! The sentence and public code a worker's failure is reported under on the
//! OpenAI-shaped surfaces.
//!
//! A streamed chat, a non-streamed chat and a Responses call each used to
//! carry their own copy of this table, which let one say "busy" where another
//! said "unavailable" for the same failure. One table keeps them identical;
//! each surface only decides how to frame the pair (an SSE error chunk, an
//! HTTP error body, a failed response).

use astronomical_ipc_protocol::{ChatGenerationFailureReason, OUT_OF_GRAPHICS_MEMORY_ADVICE};

/// The public code for a generation that ran out of graphics memory. It is
/// its own code so a client can tell this failure from an engine that is
/// simply not running, and say what helps.
pub(crate) const OUT_OF_GRAPHICS_MEMORY_CODE: &str = "chat_out_of_memory";

/// One worker failure, worded for a client of the local API.
pub(crate) struct ChatFailureWording {
    /// A sentence safe to show: bounded by the protocol, with native details
    /// left in the logs.
    pub(crate) message: String,
    /// The stable code clients match on.
    pub(crate) code: &'static str,
}

pub(crate) fn chat_failure_wording(
    failure_reason: &ChatGenerationFailureReason,
) -> ChatFailureWording {
    let (message, code) = match failure_reason {
        ChatGenerationFailureReason::ContextLengthExceeded {
            actual_total_context_tokens,
            maximum_context_tokens,
        } => (
            format!(
                "requested context uses {actual_total_context_tokens} tokens, exceeding the {maximum_context_tokens}-token model context window"
            ),
            "context_length_exceeded",
        ),
        ChatGenerationFailureReason::InvalidRequest { reason } => (
            format!("the local worker rejected the chat request: {reason}"),
            "chat_invalid_request",
        ),
        ChatGenerationFailureReason::FatalExecution { reason } => (
            format!("the local worker stopped after a fatal model execution error: {reason}"),
            "chat_worker_unavailable",
        ),
        ChatGenerationFailureReason::OutOfGraphicsMemory => (
            OUT_OF_GRAPHICS_MEMORY_ADVICE.to_owned(),
            OUT_OF_GRAPHICS_MEMORY_CODE,
        ),
        ChatGenerationFailureReason::EngineBusy => (
            "the local inference engine is already processing another request".to_owned(),
            "chat_engine_busy",
        ),
        ChatGenerationFailureReason::MalformedModelOutput => (
            "the model produced malformed structured output".to_owned(),
            "chat_malformed_model_output",
        ),
    };
    ChatFailureWording { message, code }
}
