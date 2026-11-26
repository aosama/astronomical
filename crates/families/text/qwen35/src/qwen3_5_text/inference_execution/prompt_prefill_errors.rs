//! The engine's prompt-reading errors: the shared ones, with no
//! admission refusal, because this engine reads layers as memory allows.

use std::convert::Infallible;

pub(super) use text_generation::generation_loop::prompt_prefill_errors::prefill_execution_error;

pub(super) type PromptPrefillChunkAttemptError =
    text_generation::generation_loop::prompt_prefill_errors::PromptPrefillChunkAttemptError<
        Infallible,
    >;
