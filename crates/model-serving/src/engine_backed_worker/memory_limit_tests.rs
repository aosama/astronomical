//! Unit contracts for the pure MLX memory-limit planning helpers.
//!
//! These helpers decide rejection, ceiling resolution, and image-failure
//! mapping before any protocol event is emitted, so they are covered as
//! pure units here.

use astronomical_ipc_protocol::ImageGenerationFailureReason;

use super::{
    effective_requested_ceiling_after_wired_claim, is_mlx_memory_limit_request_out_of_bounds,
    map_image_memory_limit_failure_to_engine_error,
};
use crate::InferenceEngineError;

const MACHINE_MLX_MEMORY_CEILING_BYTES: u64 = 100_000;

#[test]
fn should_reject_a_zero_byte_mlx_memory_limit_request() {
    assert!(is_mlx_memory_limit_request_out_of_bounds(
        0,
        MACHINE_MLX_MEMORY_CEILING_BYTES
    ));
}

#[test]
fn should_reject_a_request_above_the_machine_mlx_memory_ceiling() {
    assert!(is_mlx_memory_limit_request_out_of_bounds(
        MACHINE_MLX_MEMORY_CEILING_BYTES + 1,
        MACHINE_MLX_MEMORY_CEILING_BYTES
    ));
}

#[test]
fn should_accept_a_request_exactly_at_the_machine_mlx_memory_ceiling() {
    assert!(!is_mlx_memory_limit_request_out_of_bounds(
        MACHINE_MLX_MEMORY_CEILING_BYTES,
        MACHINE_MLX_MEMORY_CEILING_BYTES
    ));
}

#[test]
fn should_accept_a_request_below_the_machine_mlx_memory_ceiling() {
    assert!(!is_mlx_memory_limit_request_out_of_bounds(
        MACHINE_MLX_MEMORY_CEILING_BYTES - 1,
        MACHINE_MLX_MEMORY_CEILING_BYTES
    ));
}

#[test]
fn should_keep_the_requested_ceiling_when_no_wired_budget_is_claimed() {
    assert_eq!(
        effective_requested_ceiling_after_wired_claim(50_000, 200_000, None),
        50_000
    );
}

#[test]
fn should_lower_the_requested_ceiling_to_the_plannable_wired_residency() {
    // A 100,000-byte wired claim plans against 97,500 bytes (2.5% Metal
    // working-set margin), which sits below the 99,000-byte request.
    assert_eq!(
        effective_requested_ceiling_after_wired_claim(99_000, 200_000, Some(100_000)),
        97_500
    );
}

#[test]
fn should_cap_the_request_at_the_machine_ceiling_even_with_a_larger_wired_budget() {
    assert_eq!(
        effective_requested_ceiling_after_wired_claim(300_000, 200_000, Some(400_000)),
        200_000
    );
}

#[test]
fn should_resolve_the_requested_ceiling_at_the_machine_ceiling_boundary() {
    assert_eq!(
        effective_requested_ceiling_after_wired_claim(
            MACHINE_MLX_MEMORY_CEILING_BYTES,
            MACHINE_MLX_MEMORY_CEILING_BYTES,
            None
        ),
        MACHINE_MLX_MEMORY_CEILING_BYTES
    );
}

#[test]
fn should_map_an_engine_busy_image_failure_to_the_engine_busy_error() {
    assert!(matches!(
        map_image_memory_limit_failure_to_engine_error(ImageGenerationFailureReason::EngineBusy),
        InferenceEngineError::EngineBusy
    ));
}

#[test]
fn should_map_request_scoped_image_failures_to_fatal_errors_with_the_original_reason() {
    let invalid_request_reason = match map_image_memory_limit_failure_to_engine_error(
        ImageGenerationFailureReason::InvalidRequest {
            reason: "width out of range".to_owned(),
        },
    ) {
        InferenceEngineError::Fatal { reason } => reason,
        other => panic!("expected a fatal error, got {other:?}"),
    };
    assert_eq!(invalid_request_reason, "width out of range");

    let encoding_failure_reason = match map_image_memory_limit_failure_to_engine_error(
        ImageGenerationFailureReason::EncodingFailed {
            reason: "prompt encoding failed".to_owned(),
        },
    ) {
        InferenceEngineError::Fatal { reason } => reason,
        other => panic!("expected a fatal error, got {other:?}"),
    };
    assert_eq!(encoding_failure_reason, "prompt encoding failed");

    let fatal_execution_reason = match map_image_memory_limit_failure_to_engine_error(
        ImageGenerationFailureReason::FatalExecution {
            reason: "command buffer failed".to_owned(),
        },
    ) {
        InferenceEngineError::Fatal { reason } => reason,
        other => panic!("expected a fatal error, got {other:?}"),
    };
    assert_eq!(fatal_execution_reason, "command buffer failed");
}

#[test]
fn should_map_unrecognized_image_failures_to_a_debug_formatted_fatal_error() {
    let unrecognized_failure_reason = match map_image_memory_limit_failure_to_engine_error(
        ImageGenerationFailureReason::Cancelled,
    ) {
        InferenceEngineError::Fatal { reason } => reason,
        other => panic!("expected a fatal error, got {other:?}"),
    };
    assert_eq!(
        unrecognized_failure_reason,
        "image memory-limit update failed: Cancelled"
    );
}
