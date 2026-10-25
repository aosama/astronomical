//! A separate 37-gigabyte diagnostic cell, not a change to the 36-gigabyte sweep.

use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use serde_json::json;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use super::{
    historical_record::ThroughputJourneyKind,
    support::{self, ThroughputJourney},
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the sparse model at 37 GB for a warmed 26552-input/500-output full-fit residency diagnostic"]
#[serial]
async fn should_complete_decode_residency_at_37_gigabytes_after_multi_chunk_prefill() {
    let model_id = crate::support::large_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let measured_prompt = crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        &model_directory,
        include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt"),
        "Continue the supplied Romeo and Juliet story for approximately 500 tokens.",
        26_552,
    );
    let journey = ThroughputJourney {
        journey_kind: ThroughputJourneyKind::DecodeFullFit,
        decode_expert_cache_policy: WorkerDecodeExpertCachePolicy::Baseline,
        warmup_input_prompt: format!(
            "Continue Romeo and Juliet.\n\n{}",
            include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt")
        ),
        warmup_images: Vec::new(),
        warmup_output_tokens: 50,
        measured_input_prompt: measured_prompt,
        measured_images: Vec::new(),
        measured_output_tokens: 500,
        temperature_thousandths: 1_000,
        maximum_mlx_memory_bytes: Some(sparse_decode_full_fit::MEMORY_CEILING_BYTES),
        attribution_enabled: true,
        journey_timeout: Duration::from_secs(110),
        model_directory: Some(model_directory),
        diagnostic_qwen3_5_engine_override:
            astronomical_ipc_protocol::WorkerDiagnosticQwen3_5EngineOverride::Automatic,
        diagnostic_qwen3_5_resident_prefill_activation_floor_bytes: None,
    };
    let outcome = timeout(
        Duration::from_secs(115),
        support::measure_throughput(
            crate::support::large_sparse_moe_streaming_model_id(),
            &journey,
        ),
    )
    .await
    .expect("the full-fit performance diagnostic must remain inside the 120-second runner budget");
    assert!(
        !outcome.measured_budget_exhausted,
        "the diagnostic exhausted its bounded journey"
    );
    assert!(
        (26_500..=26_552).contains(&outcome.measurement.total_input_tokens),
        "the diagnostic must reproduce the long-context admission boundary"
    );
    assert!((450..=550).contains(&outcome.measurement.total_output_tokens));
    assert_eq!(outcome.measurement.cached_tokens, 0);
    let report = sparse_decode_full_fit::read_generation_attribution_report(
        &outcome.logging_directory,
        support::MEASURED_REQUEST_ID,
    )
    .expect("the measured request must have attribution");
    let snapshot = &outcome.final_health_snapshot;
    let status = json!({
        "mlx_memory_ceiling_bytes": snapshot.mlx_memory_ceiling_bytes,
        "expert_memory_mode": snapshot.expert_memory_mode,
        "expert_residency": snapshot.expert_residency.as_ref().map(|residency| json!({
            "total_layer_count": residency.total_layer_count,
            "resident_expert_count": residency.resident_expert_count,
            "resident_expert_payload_bytes": residency.resident_expert_payload_bytes
        })),
        "mlx_memory_snapshot": snapshot.latest_mlx_memory_snapshot
    });
    std::fs::write(
        outcome
            .logging_directory
            .join("decode-full-fit-evidence.json"),
        serde_json::to_vec_pretty(&json!({
            "status": status,
            "attribution": report,
            "input_tokens": outcome.measurement.total_input_tokens,
            "output_tokens": outcome.measurement.total_output_tokens,
            "prefill_tokens_per_second": outcome.measurement.prefill_tokens_per_second,
            "decode_tokens_per_second": outcome.measurement.decode_tokens_per_second
        }))
        .expect("the diagnostic evidence should serialize"),
    )
    .expect("the diagnostic evidence must persist");
    sparse_decode_full_fit::validate_decode_residency(&status).unwrap_or_else(|failure| {
        panic!(
            "{failure}; evidence={}",
            outcome.logging_directory.display()
        )
    });
    sparse_decode_full_fit::validate_generation_attribution(&report).unwrap_or_else(|failure| {
        panic!(
            "{failure}; evidence={}",
            outcome.logging_directory.display()
        )
    });
}
