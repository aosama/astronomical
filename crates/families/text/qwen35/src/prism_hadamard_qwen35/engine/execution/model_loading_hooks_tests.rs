use serial_test::serial;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::memory::dense_resident_safe_minimum_mlx_memory_ceiling_bytes;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::engine::PrismHadamardQwen35InferenceExecution;
use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    FixtureMemoryLimits, pending_fixture_execution, serving_settings_with,
};
use crate::{InferenceEngineError, MlxInferenceExecution, PrismHadamardQwen35ServingSettings};

const FIXTURE_MEMORY_LIMITS: FixtureMemoryLimits = FixtureMemoryLimits {
    mlx_memory_ceiling_bytes: 512 * 1024 * 1024,
    mlx_buffer_cache_bytes: 512 * 1024 * 1024,
};

/// Loads an engine over a fresh fixture pack with `serving_settings`,
/// returning the pack's payload size, the engine and what its first load
/// reported. The pack directory is removed before returning: the model has
/// mapped its weights by then, and a failed load never needs it again.
fn load_fixture_execution(
    serving_settings: &PrismHadamardQwen35ServingSettings,
) -> (
    u64,
    PrismHadamardQwen35InferenceExecution,
    Result<EngineLoadResult, InferenceEngineError>,
) {
    let fixture_directory = write_fixture_pack(|_| {});
    let mut execution = pending_fixture_execution(
        &fixture_directory,
        serving_settings,
        &FIXTURE_MEMORY_LIMITS,
        None,
    );
    let pack_payload_bytes = execution
        .execution_core
        .validated_artifact
        .as_ref()
        .expect("a pending engine holds its validated pack")
        .total_payload_bytes();
    let first_load = execution.load();
    remove_fixture_directory(&fixture_directory);
    (pack_payload_bytes, execution, first_load)
}

// Positive: the load reports the whole pack resident, the dense floor as
// the minimum ceiling, and a Neural Engine prefill it cannot serve.
#[test]
#[serial]
fn should_report_the_dense_floor_and_resident_placement_after_a_load() {
    let (pack_payload_bytes, execution, first_load) =
        load_fixture_execution(&serving_settings_with(64, 1));

    let load_result = first_load.expect("the fixture pack should load");
    assert_eq!(
        load_result.expert_memory_mode(),
        Some(ipc_protocol::ExpertMemoryMode::Resident)
    );
    assert_eq!(
        load_result.minimum_mlx_memory_ceiling_bytes(),
        dense_resident_safe_minimum_mlx_memory_ceiling_bytes(pack_payload_bytes)
    );
    assert!(matches!(
        load_result.neural_engine_prefill_event(),
        Some(ipc_protocol::WorkerEvent::NeuralEnginePrefillChanged {
            neural_engine_prefill: ipc_protocol::NeuralEnginePrefillStatus::Unavailable { .. }
        })
    ));
    let execution_core = &execution.execution_core;
    assert!(execution_core.model.is_some());
    assert!(execution_core.validated_artifact.is_none());
    assert!(execution_core.model_id.is_some() && execution_core.model_revision.is_some());
}

// Edge: a second load reports the model already held instead of failing,
// and a load that never asked for the Neural Engine announces nothing.
#[test]
#[serial]
fn should_report_the_held_model_on_a_second_load() {
    let mut serving_settings = serving_settings_with(64, 1);
    serving_settings
        .chunking
        .experimental_neural_engine_prefill_enabled = false;
    let (_pack_payload_bytes, mut execution, first_load) =
        load_fixture_execution(&serving_settings);
    let first_load_result = first_load.expect("the fixture pack should load");

    let second_load_result = execution
        .load()
        .expect("a second load reports the model already held");

    assert_eq!(second_load_result, first_load_result);
    assert!(first_load_result.neural_engine_prefill_event().is_none());
}

// Negative: chunking settings the shared load refuses fail the load as an
// engine failure, and no model is installed.
#[test]
#[serial]
fn should_fail_a_load_whose_chunking_settings_do_not_validate() {
    let mut serving_settings = serving_settings_with(64, 1);
    serving_settings
        .chunking
        .full_attention_key_value_growth_tokens = 0;
    let (_pack_payload_bytes, execution, first_load) = load_fixture_execution(&serving_settings);

    assert!(matches!(
        first_load,
        Err(InferenceEngineError::Fatal { .. })
    ));
    assert!(execution.execution_core.model.is_none());
}
