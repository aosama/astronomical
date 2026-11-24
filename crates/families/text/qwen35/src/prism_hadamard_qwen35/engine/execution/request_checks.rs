//! The checks a folded-ternary request must pass before it may start.
//!
//! The resident and streaming engines refuse a prompt with a token outside
//! the vocabulary, or one whose prompt and output budget cannot fit the
//! context window, before any state is built. The folded-ternary engine
//! makes the same refusals, so a request that could never finish is turned
//! away at the start instead of failing part way through its reply.

use crate::InferenceEngineError;

/// Refuses a request the model cannot serve: an empty prompt, a prompt token
/// outside the vocabulary, or a prompt and output budget beyond the context
/// window.
pub(super) fn check_serving_request(
    prompt_token_ids: &[u32],
    maximum_output_tokens: u16,
    vocabulary_size: u32,
    maximum_context_tokens: u32,
) -> Result<(), InferenceEngineError> {
    if prompt_token_ids.is_empty() {
        return Err(invalid_request("folded-ternary prompt must contain tokens"));
    }
    if prompt_token_ids
        .iter()
        .any(|prompt_token_id| *prompt_token_id >= vocabulary_size)
    {
        return Err(invalid_request(
            "folded-ternary prompt contains a token outside the model vocabulary",
        ));
    }
    let requested_context_tokens = prompt_token_ids
        .len()
        .saturating_add(usize::from(maximum_output_tokens));
    if requested_context_tokens > maximum_context_tokens as usize {
        return Err(invalid_request(
            "folded-ternary prompt and output budget exceed the model context window",
        ));
    }
    Ok(())
}

fn invalid_request(reason: &str) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
#[path = "request_checks_tests.rs"]
mod request_checks_tests;
