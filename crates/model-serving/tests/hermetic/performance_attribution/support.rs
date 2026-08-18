use super::*;

pub(super) fn serialize_model_loading_report(
    performance_attribution: PerformanceAttribution,
    performance_attribution_outcome: PerformanceAttributionOutcome,
) -> Value {
    let performance_attribution_report = performance_attribution
        .finish_model_loading(model_loading_metadata(performance_attribution_outcome))
        .expect("enabled attribution should produce one model-loading report");
    serde_json::to_value(performance_attribution_report)
        .expect("model-loading attribution should serialize")
}

pub(super) fn serialize_generation_report(
    performance_attribution: PerformanceAttribution,
    performance_attribution_outcome: PerformanceAttributionOutcome,
) -> Value {
    let performance_attribution_report = performance_attribution
        .finish_generation(generation_metadata(performance_attribution_outcome))
        .expect("enabled attribution should produce one generation report");
    serde_json::to_value(performance_attribution_report)
        .expect("generation attribution should serialize")
}

pub(super) fn generation_metadata(
    performance_attribution_outcome: PerformanceAttributionOutcome,
) -> GenerationPerformanceAttributionMetadata {
    GenerationPerformanceAttributionMetadata {
        outcome: performance_attribution_outcome,
        model_id: "Qwen3.6-35B-A3B-OptiQ-4bit".to_owned(),
        model_revision: "revision".to_owned(),
        drafter_model_id: None,
        drafter_model_revision: None,
        drafter_storage_fingerprint: None,
        prefill_transient_observation_completed: true,
        prefill_observed_transient_high_water_bytes: ATTRIBUTED_PREFILL_TRANSIENT_HIGH_WATER_BYTES,
        request_id: 42,
        configured_maximum_output_tokens: 512,
        mlx_active_memory_bytes: Some(1),
        mlx_allocator_cache_memory_bytes: Some(2),
        mlx_peak_memory_bytes: Some(3),
        failure_description: Some("simulated generation failure".to_owned()),
    }
}

pub(super) fn model_loading_metadata(
    performance_attribution_outcome: PerformanceAttributionOutcome,
) -> ModelLoadingPerformanceAttributionMetadata {
    ModelLoadingPerformanceAttributionMetadata {
        outcome: performance_attribution_outcome,
        model_id: Some("Qwen3.6-35B-A3B-OptiQ-4bit".to_owned()),
        model_revision: Some("revision".to_owned()),
        drafter_model_id: None,
        drafter_model_revision: None,
        drafter_storage_fingerprint: None,
        prefill_transient_observation_completed: false,
        prefill_observed_transient_high_water_bytes: 0,
        total_artifact_payload_bytes: Some(22_135_339_264),
        resident_model_payload_bytes: Some(2_539_550_976),
        model_shard_count: Some(5),
        mlx_active_memory_bytes: Some(2_539_550_976),
        mlx_allocator_cache_memory_bytes: Some(0),
        mlx_peak_memory_bytes: Some(2_539_550_976),
        failure_description: None,
    }
}
