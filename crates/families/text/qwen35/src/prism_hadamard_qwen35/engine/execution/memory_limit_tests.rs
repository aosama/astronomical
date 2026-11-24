use super::{
    ceiling_change_refusal, memory_limits_for_ceiling, mlx_buffer_cache_bytes_for_ceiling,
};
use crate::InferenceEngineError;

const ONE_GB: u64 = 1_073_741_824;

// Positive: an idle engine accepts a ceiling at or above its minimum.
#[test]
fn should_accept_a_ceiling_change_on_an_idle_engine() {
    assert!(ceiling_change_refusal(false, 12 * ONE_GB, 10 * ONE_GB).is_none());
}

// Edge: a ceiling exactly at the minimum is accepted.
#[test]
fn should_accept_a_ceiling_exactly_at_the_minimum() {
    assert!(ceiling_change_refusal(false, 10 * ONE_GB, 10 * ONE_GB).is_none());
}

// Negative: a change while a request runs waits, as on the resident engine.
#[test]
fn should_refuse_a_ceiling_change_while_a_request_runs() {
    let refusal = ceiling_change_refusal(true, 12 * ONE_GB, 10 * ONE_GB);

    assert!(matches!(refusal, Some(InferenceEngineError::EngineBusy)));
}

// Negative: one byte below the minimum is the typed rejection.
#[test]
fn should_reject_a_ceiling_one_byte_below_the_minimum() {
    let refusal = ceiling_change_refusal(false, 10 * ONE_GB - 1, 10 * ONE_GB);

    assert!(matches!(
        refusal,
        Some(InferenceEngineError::MlxMemoryLimitRejected {
            requested_mlx_memory_ceiling_bytes,
            minimum_mlx_memory_ceiling_bytes,
            ..
        }) if requested_mlx_memory_ceiling_bytes == 10 * ONE_GB - 1
            && minimum_mlx_memory_ceiling_bytes == 10 * ONE_GB
    ));
}

// Positive: a raised ceiling restores the configured mlx_buffer_cache limit.
#[test]
fn should_restore_the_configured_buffer_limit_when_the_ceiling_rises_again() {
    let lowered = mlx_buffer_cache_bytes_for_ceiling(2 * ONE_GB, ONE_GB);
    let raised = mlx_buffer_cache_bytes_for_ceiling(2 * ONE_GB, 20 * ONE_GB);

    assert_eq!(lowered, ONE_GB);
    assert_eq!(raised, 2 * ONE_GB);
}

// Edge: a ceiling equal to the configured limit keeps it unchanged.
#[test]
fn should_keep_the_configured_buffer_limit_at_an_equal_ceiling() {
    assert_eq!(
        mlx_buffer_cache_bytes_for_ceiling(2 * ONE_GB, 2 * ONE_GB),
        2 * ONE_GB
    );
}

// Edge: before the model loads the minimum is one byte, so a zero ceiling is
// still the typed rejection rather than a silently raised one.
#[test]
fn should_reject_a_zero_ceiling_before_the_model_loads() {
    let refusal = ceiling_change_refusal(false, 0, 1);

    assert!(matches!(
        refusal,
        Some(InferenceEngineError::MlxMemoryLimitRejected { .. })
    ));
}

// Positive: an accepted ceiling becomes the MLX limits the core keeps.
#[test]
fn should_build_the_limits_of_an_accepted_ceiling() {
    let memory_limits = memory_limits_for_ceiling(24 * ONE_GB, 2 * ONE_GB)
        .expect("a ceiling above its mlx_buffer_cache limit must validate");
    assert_eq!(
        memory_limits.active_memory_limit_bytes() as u64,
        24 * ONE_GB
    );
    assert_eq!(
        memory_limits.mlx_buffer_cache_limit_bytes() as u64,
        2 * ONE_GB
    );
}

// Edge: an mlx_buffer_cache limit capped down to the ceiling still validates.
#[test]
fn should_accept_an_mlx_buffer_cache_limit_equal_to_the_ceiling() {
    let capped_mlx_buffer_cache_bytes = mlx_buffer_cache_bytes_for_ceiling(8 * ONE_GB, 4 * ONE_GB);
    assert!(memory_limits_for_ceiling(4 * ONE_GB, capped_mlx_buffer_cache_bytes).is_ok());
}

// Negative: limits the accepted-ceiling rule never produces are refused.
#[test]
fn should_refuse_limits_that_break_the_ceiling_rule() {
    assert!(matches!(
        memory_limits_for_ceiling(0, 0),
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(matches!(
        memory_limits_for_ceiling(ONE_GB, 2 * ONE_GB),
        Err(InferenceEngineError::Fatal { .. })
    ));
}
