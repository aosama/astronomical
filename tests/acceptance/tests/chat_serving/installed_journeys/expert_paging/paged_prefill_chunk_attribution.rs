//! Per-chunk attribution of paged prompt prefill at the model layer.
//!
//! Bottom-up probe, one level above the storage boundary: the production
//! `Qwen3_5StreamingModel` executes real 2,048-token Romeo and Juliet prefill
//! chunks under the artifact's derived paging ceiling (SSD streaming active,
//! no resident complete layers), and each chunk receives its own
//! `PerformanceAttribution` so the probe reports, per chunk:
//!
//! - logical expert-stream payload bytes versus the whole-model expert payload
//!   (the re-stream multiplier: how many times the chunk pulled expert bytes
//!   that equal the complete sparse payload),
//! - positional read calls, bytes, and cumulative read time,
//! - time split across expert-stream preparation, materialization waits,
//!   graph construction, and graphics-processor waits.
//!
//! The journey asserts only shape (chunks complete, experts actually streamed,
//! reads actually issued); the printed table is the diagnostic evidence.

use std::time::Duration;

use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, PerformanceAttribution, PerformanceCounter,
    PerformanceOperation, Qwen3_5ArtifactValidator, Qwen3_5StreamingModel,
    mlx_ram_budget_model_geometry_from_validated_artifact,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;
use tokio::time::timeout;

use super::prompt;

const PROMPT_TOKEN_COUNT: usize = 4_097;
const OUTPUT_TOKEN_COUNT: u16 = 8;
const PREFILL_CHUNK_TOKENS: usize = 2_048;
const PAGED_PREFILL_ATTRIBUTION_TIMEOUT: Duration = Duration::from_secs(60);

#[tokio::test]
#[ignore = "loads the configured sparse model under its paging ceiling and attributes each paged prefill chunk"]
#[serial]
async fn should_attribute_expert_streaming_per_paged_prefill_chunk() {
    timeout(
        PAGED_PREFILL_ATTRIBUTION_TIMEOUT,
        attribute_paged_prefill_chunks(),
    )
    .await
    .expect("the paged prefill attribution journey must finish within 60 seconds");
}

async fn attribute_paged_prefill_chunks() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, u32::from(OUTPUT_TOKEN_COUNT))
        .expect("the configured sparse artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let (model_geometry, required_headroom_bytes) =
        mlx_ram_budget_model_geometry_from_validated_artifact(
            &validated_artifact,
            &model_directory,
        )
        .expect("the sparse artifact should expose disk RAM geometry");
    let residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    );
    let paging_ceiling_bytes = residency_boundary
        .paging_ceiling_bytes()
        .expect("this sparse artifact must have a paging ceiling below complete residency");
    eprintln!(
        "[paged-prefill-attribution] status=progress phase=derived_ceiling complete_expert_payload_bytes={} headroom_bytes={required_headroom_bytes} paging_ceiling_bytes={paging_ceiling_bytes}",
        model_geometry.complete_expert_payload_bytes
    );

    let prompt_token_ids =
        prompt::prepare_reproduced_long_prompt_token_ids(PROMPT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT);
    assert!(
        prompt_token_ids.len() >= PROMPT_TOKEN_COUNT,
        "the reproduced Romeo and Juliet prompt must cover at least two full chunks"
    );

    let runtime = MlxRuntime::initialize(
        astronomical_runtime_integration::MlxMemoryLimits::new(
            usize::try_from(paging_ceiling_bytes).expect("the paging ceiling should fit usize"),
            usize::try_from(paging_ceiling_bytes).expect("the paging ceiling should fit usize"),
        )
        .expect("the paging ceiling should be a valid MLX limit"),
    )
    .expect("the attributed runtime should initialize");
    eprintln!("[paged-prefill-attribution] status=progress phase=model_load ETA_seconds=30");
    let qwen3_5_model = Qwen3_5StreamingModel::load(
        runtime,
        validated_artifact,
        &model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the configured model should load under the paging ceiling");
    let mut request_decoder_state = crate::common::standard_request_decoder_state(&qwen3_5_config);

    // Mirror the engine's request-open publication: the composed Prefill
    // leftover budget raises the retained-expert ceiling and freezes the
    // request-stable pin/stream contract before the first chunk executes.
    let mut request_open_attribution = PerformanceAttribution::enabled();
    qwen3_5_model
        .refresh_phase_aware_expert_residency_plan_for_tests(
            astronomical_model_serving::MemoryPhase::Prefill,
            u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
            u64::try_from(PREFILL_CHUNK_TOKENS).unwrap_or(u64::MAX),
            &mut request_open_attribution,
        )
        .expect("the request-open residency plan should publish");
    let opening_cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
    eprintln!(
        "[paged-prefill-attribution] request_open cache_ceiling_gb={:.3}",
        opening_cache_statistics.maximum_resident_payload_byte_count as f64 / 1e9
    );

    let mut total_positional_read_bytes = 0_u64;
    let mut total_streamed_payload_bytes = 0_u64;
    let final_prompt_token_index = prompt_token_ids.len() - 1;
    let mut chunk_index = 0_usize;
    for chunk_start in (0..final_prompt_token_index).step_by(PREFILL_CHUNK_TOKENS) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_TOKENS).min(final_prompt_token_index);
        chunk_index += 1;
        // The engine re-publishes the residency plan before every chunk advance
        // (context position grows, learned reserves change); replicating that
        // here keeps the probe faithful to serving behavior.
        qwen3_5_model
            .refresh_phase_aware_expert_residency_plan_for_tests(
                astronomical_model_serving::MemoryPhase::Prefill,
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
                u64::try_from(chunk_end - chunk_start).unwrap_or(u64::MAX),
                &mut PerformanceAttribution::enabled(),
            )
            .expect("the per-chunk residency plan refresh should publish");
        let chunk_started_at = std::time::Instant::now();
        let mut chunk_attribution = PerformanceAttribution::enabled();
        qwen3_5_model
            .prefill_chunk_with_attribution_for_tests(
                &prompt_token_ids[chunk_start..chunk_end],
                u32::try_from(chunk_start).expect("the diagnostic prompt position should fit u32"),
                &mut request_decoder_state,
                &mut chunk_attribution,
            )
            .expect("the attributed paged prefill chunk should complete");
        let chunk_elapsed = chunk_started_at.elapsed();

        let streamed_payload_bytes = chunk_attribution
            .counter_value(PerformanceCounter::RustExpertStreamingPayloadByteCount);
        let seated_payload_bytes = chunk_attribution.counter_value(
            PerformanceCounter::ExpertResidencyReadThroughSeatedCompletePayloadBytes,
        );
        let full_hit_assignments =
            chunk_attribution.counter_value(PerformanceCounter::RetainedRouteAssignmentHitCount);
        let (read_calls, read_bytes, read_elapsed_ns) = chunk_attribution
            .positional_file_read_metrics()
            .map(|metrics| metrics.snapshot())
            .map(|snapshot| {
                (
                    snapshot.read_call_count,
                    snapshot.read_byte_count,
                    snapshot.total_read_elapsed_nanoseconds,
                )
            })
            .unwrap_or((0, 0, 0));
        total_positional_read_bytes += read_bytes;
        total_streamed_payload_bytes += streamed_payload_bytes;

        let stream_prep = operation_summary(
            &chunk_attribution,
            PerformanceOperation::RustExpertStreamingLayerPreparation,
        );
        let materialization_wait = operation_summary(
            &chunk_attribution,
            PerformanceOperation::MandatoryPrefillCompleteLayerMaterializationWait,
        );
        let residency_planning = operation_summary(
            &chunk_attribution,
            PerformanceOperation::ExpertResidencyPlanning,
        );
        let admission = operation_summary(
            &chunk_attribution,
            PerformanceOperation::MemoryAdmissionSnapshot,
        );
        let moe_graph = operation_summary(
            &chunk_attribution,
            PerformanceOperation::PagedMoeGraphConstruction,
        );
        let mixture_wait = operation_summary(
            &chunk_attribution,
            PerformanceOperation::PrefillMixtureOfValuesGraphicsProcessorCompletionWait,
        );
        eprintln!(
            "[paged-prefill-attribution] chunk={chunk_index} tokens={} wall_ms={} streamed_gb={:.3} seated_gb={:.3} full_hit_assignments={full_hit_assignments} read_calls={read_calls} read_gb={:.3} read_ms={:.0} | ms: stream_prep={stream_prep} materialize_wait={materialization_wait} residency_plan={residency_planning} admission={admission} moe_graph={moe_graph} mixture_gpu_wait={mixture_wait}",
            chunk_end - chunk_start,
            chunk_elapsed.as_millis() as f64,
            streamed_payload_bytes as f64 / 1e9,
            seated_payload_bytes as f64 / 1e9,
            read_bytes as f64 / 1e9,
            read_elapsed_ns as f64 / 1e6,
        );
        assert!(
            streamed_payload_bytes > 0,
            "under the paging ceiling every chunk must stream expert pages"
        );
        assert!(
            read_bytes > 0,
            "every streamed chunk must issue positional reads"
        );
        let cache_statistics = qwen3_5_model.expert_weight_memory_cache_statistics();
        eprintln!(
            "[paged-prefill-attribution] chunk={chunk_index} cache_after: entries={} resident_gb={:.3} complete_layers={} evictions={} max_gb={:.3}",
            cache_statistics.entry_count,
            cache_statistics.resident_payload_byte_count as f64 / 1e9,
            cache_statistics.complete_layer_count,
            cache_statistics.eviction_count,
            cache_statistics.maximum_resident_payload_byte_count as f64 / 1e9,
        );
    }

    let complete_expert_payload_bytes = model_geometry.complete_expert_payload_bytes;
    let restream_multiplier =
        total_streamed_payload_bytes as f64 / complete_expert_payload_bytes.max(1) as f64;
    eprintln!(
        "[paged-prefill-attribution] summary chunks={chunk_index} total_streamed_gb={:.3} total_read_gb={:.3} complete_expert_payload_gb={:.3} logical_restream_multiplier={restream_multiplier:.2}",
        total_streamed_payload_bytes as f64 / 1e9,
        total_positional_read_bytes as f64 / 1e9,
        complete_expert_payload_bytes as f64 / 1e9,
    );
}

fn operation_summary(
    attribution: &PerformanceAttribution,
    operation: PerformanceOperation,
) -> String {
    match attribution.operation_measurement(operation) {
        Some(measurement) => format!(
            "{}x{}ms",
            measurement.occurrence_count(),
            measurement.total_elapsed_nanoseconds() as f64 / 1e6
        ),
        None => "0".to_owned(),
    }
}
