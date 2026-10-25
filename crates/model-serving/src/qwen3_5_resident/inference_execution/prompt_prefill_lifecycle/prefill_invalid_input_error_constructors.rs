//! Named invalid-input execution error constructors for the prefill phase.
//!
//! The forward dispatch and terminal seed reject malformed internal state with
//! `Qwen3_5ExecutionError::InvalidInput`. Every function here exists so call
//! sites pass a name by path instead of writing a closure, and the tests lock
//! each description byte-for-byte so replacing the inline closures stays
//! provably behavior-preserving.

use crate::Qwen3_5ExecutionError;

pub(super) fn engine_lost_loaded_model_invalid_input() -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput {
        description: "Qwen3.5 engine lost its loaded model",
    }
}

pub(super) fn visual_prompt_cache_checkpoints_have_no_block_contract_invalid_input()
-> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput {
        description: "planned visual prompt-cache checkpoints have no block contract",
    }
}

pub(super) fn text_prompt_cache_checkpoints_have_no_block_contract_invalid_input()
-> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput {
        description: "planned text prompt-cache checkpoints have no block contract",
    }
}

/// The terminal prefill seed samples the first generated token from the final
/// prompt logits; a sampler failure here is unrecoverable for the request.
pub(super) fn terminal_prefill_first_generated_token_sampling_failed_invalid_input<
    SamplingError,
>(
    _sampling_error: SamplingError,
) -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput {
        description: "failed to sample the first generated token from terminal prefill logits",
    }
}

#[cfg(test)]
#[path = "prefill_invalid_input_error_constructors_tests.rs"]
mod prefill_invalid_input_error_constructors_tests;
