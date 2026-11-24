//! The streaming engine's prompt-reading errors: the shared ones, with no
//! admission refusal, because this engine reads layers as memory allows.

use std::convert::Infallible;

pub(super) use crate::qwen3_5_core::shared_execution::prompt_prefill_errors::prefill_execution_error;

pub(super) type PromptPrefillChunkAttemptError =
    crate::qwen3_5_core::shared_execution::prompt_prefill_errors::PromptPrefillChunkAttemptError<
        Infallible,
    >;
