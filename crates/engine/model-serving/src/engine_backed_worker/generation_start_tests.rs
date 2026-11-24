//! Beside-source coverage for the pure generation-start units: the protocol
//! prompt-token-count conversion. The promotion gate, the streaming retry,
//! and the startup event emission run under the worker test doubles.

use super::protocol_prompt_token_count;
use crate::engine_backed_worker::support::WorkerRuntimeError;

#[test]
fn should_convert_an_in_range_prompt_token_count() {
    let outcome = protocol_prompt_token_count(151_936);

    assert!(matches!(outcome, Ok(151_936)));
}

#[test]
fn should_convert_the_u32_ceiling_prompt_token_count() {
    let outcome = protocol_prompt_token_count(u32::MAX as usize);

    assert!(matches!(outcome, Ok(generated) if generated == u32::MAX));
}

#[test]
fn should_reject_a_prompt_token_count_beyond_the_protocol_range() {
    let outcome = protocol_prompt_token_count(u32::MAX as usize + 1);
    let generation_error =
        outcome.expect_err("a u32-exceeding prompt must fail the protocol conversion");

    assert!(
        matches!(
            &generation_error,
            WorkerRuntimeError::InferenceEngineGenerationFailed { reason }
                if reason == "chat prompt token count exceeds the protocol range"
        ),
        "the failure keeps the protocol-range reason: {generation_error:?}"
    );
}
