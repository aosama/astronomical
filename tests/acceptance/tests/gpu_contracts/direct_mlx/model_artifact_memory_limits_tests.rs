//! Contracts for the acceptance harness's MLX memory-ceiling resolution.
//!
//! The configured user ceiling must never raise the effective limit above the
//! machine ceiling, and the machine ceiling applies verbatim when no user
//! limit exists. These run in the direct-MLX contract binary because the
//! helper they pin lives in this package's shared GPU-contract module.

use crate::common::resolve_serving_acceptance_mlx_memory_ceiling_bytes;

#[test]
fn should_use_the_configured_model_artifact_memory_ceiling_when_it_is_lower_than_the_machine() {
    assert_eq!(
        resolve_serving_acceptance_mlx_memory_ceiling_bytes(Some(35_000_000_000), 40_000_000_000),
        35_000_000_000,
    );
}

#[test]
fn should_never_raise_model_artifact_memory_above_the_machine_ceiling() {
    assert_eq!(
        resolve_serving_acceptance_mlx_memory_ceiling_bytes(Some(45_000_000_000), 40_000_000_000),
        40_000_000_000,
    );
}

#[test]
fn should_use_the_machine_model_artifact_memory_ceiling_when_no_user_limit_exists() {
    assert_eq!(
        resolve_serving_acceptance_mlx_memory_ceiling_bytes(None, 40_000_000_000),
        40_000_000_000,
    );
}
