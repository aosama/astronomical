use ipc_protocol::{ExpertMemoryMode, RequestId};
use serial_test::file_serial;

use super::{
    ceiling_change_refusal, memory_limits_for_ceiling, mlx_buffer_cache_bytes_for_ceiling,
};
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    FixtureMemoryLimits, loaded_fixture_engine, pending_fixture_execution, serving_settings_with,
};
use crate::{InferenceEngineError, MlxInferenceExecution, TextInferenceRequest};
use text_generation::model::text_model::TextModel;

const ONE_GB: u64 = 1_073_741_824;

// Positive: a ceiling above the minimum is accepted.
#[test]
fn should_accept_a_ceiling_above_the_minimum() {
    assert!(ceiling_change_refusal(12 * ONE_GB, 10 * ONE_GB).is_none());
}

// Edge: a ceiling exactly at the minimum is accepted.
#[test]
fn should_accept_a_ceiling_exactly_at_the_minimum() {
    assert!(ceiling_change_refusal(10 * ONE_GB, 10 * ONE_GB).is_none());
}

// Negative: one byte below the minimum is the typed rejection.
#[test]
fn should_reject_a_ceiling_one_byte_below_the_minimum() {
    let refusal = ceiling_change_refusal(10 * ONE_GB - 1, 10 * ONE_GB);

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

// Edge: a zero ceiling is the typed rejection rather than a silently raised
// one, even against the smallest minimum.
#[test]
fn should_reject_a_zero_ceiling() {
    let refusal = ceiling_change_refusal(0, 1);

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

// Positive: an idle loaded engine moves to a ceiling at its minimum and
// back up, keeping the configured mlx_buffer_cache limit under both, and
// reports the memory after each change.
#[test]
#[file_serial]
fn should_change_the_ceiling_of_an_idle_engine_and_report_the_memory_after_it() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    let minimum_mlx_memory_ceiling_bytes = fixture_engine
        .execution
        .execution_core
        .loaded_model()
        .expect("the fixture model is loaded")
        .minimum_mlx_memory_ceiling_bytes()
        .expect("the dense floor is known");

    for requested_mlx_memory_ceiling_bytes in [
        minimum_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes + ONE_GB,
    ] {
        let adjustment = fixture_engine
            .execution
            .update_mlx_memory_limit(requested_mlx_memory_ceiling_bytes)
            .expect("a ceiling at or above the floor should apply");
        assert_eq!(
            adjustment.effective_mlx_memory_ceiling_bytes(),
            requested_mlx_memory_ceiling_bytes
        );
        // The fixture worker configured 512 MiB, below every accepted ceiling.
        assert_eq!(adjustment.mlx_buffer_cache_limit_bytes(), 512 * 1024 * 1024);
        assert_eq!(
            adjustment.minimum_mlx_memory_ceiling_bytes(),
            minimum_mlx_memory_ceiling_bytes
        );
        assert_eq!(adjustment.expert_memory_mode(), ExpertMemoryMode::Resident);
        assert!(adjustment.mlx_memory_telemetry().is_some());
        assert!(adjustment.expert_residency_telemetry().is_some());
        assert_eq!(
            fixture_engine
                .execution
                .execution_core
                .memory_limits
                .active_memory_limit_bytes() as u64,
            requested_mlx_memory_ceiling_bytes
        );
    }
}

// Negative: a change waits while a request runs, and a ceiling below the
// floor is the typed rejection; neither moves the installed limits.
#[test]
#[file_serial]
fn should_refuse_a_change_while_a_request_runs_or_below_the_floor() {
    let mut fixture_engine = loaded_fixture_engine(|_| {});
    let installed_mlx_memory_limits = fixture_engine.execution.execution_core.memory_limits;
    fixture_engine
        .execution
        .start_generation(TextInferenceRequest::new_sampling(
            RequestId::new(61),
            vec![11, 12, 13],
            4,
            0,
            1_000,
            None,
        ))
        .expect("the request should start");
    assert!(matches!(
        fixture_engine
            .execution
            .update_mlx_memory_limit(64 * ONE_GB),
        Err(InferenceEngineError::EngineBusy)
    ));
    fixture_engine
        .execution
        .cancel_generation(RequestId::new(61))
        .expect("the request should cancel");

    assert!(matches!(
        fixture_engine.execution.update_mlx_memory_limit(ONE_GB - 1),
        Err(InferenceEngineError::MlxMemoryLimitRejected { .. })
    ));
    assert_eq!(
        fixture_engine.execution.execution_core.memory_limits,
        installed_mlx_memory_limits
    );
}

// Negative: a change before any model is loaded is an engine failure, as
// on every Qwen 3.5 engine.
#[test]
#[file_serial]
fn should_refuse_a_change_before_the_model_loads() {
    let fixture_directory = write_fixture_pack(|_| {});
    let mut execution = pending_fixture_execution(
        &fixture_directory,
        &serving_settings_with(64, 1),
        &FixtureMemoryLimits {
            mlx_memory_ceiling_bytes: 512 * 1024 * 1024,
            mlx_buffer_cache_bytes: 512 * 1024 * 1024,
        },
        None,
    );

    let refusal = execution.update_mlx_memory_limit(64 * ONE_GB);

    remove_fixture_directory(&fixture_directory);
    assert!(matches!(refusal, Err(InferenceEngineError::Fatal { .. })));
}
