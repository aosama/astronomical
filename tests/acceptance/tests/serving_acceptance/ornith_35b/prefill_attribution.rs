//! Ornith-1.5-35B prefill graphics-processor attribution by op family.
//!
//! This journey isolates where prompt-processing wall time is spent on the
//! graphics processor for the resident 35B sparse-MoE model. It runs one
//! short excerpt warmup that absorbs the model load and first-use JIT kernel
//! compilation, then one measured completion over the full Romeo and Juliet
//! prompt against a cold prompt cache, with performance attribution enabled in
//! the isolated home. The worker's per-layer prefill evaluation boundaries
//! then attribute the prefill graphics-processor time to the linear-attention
//! family, the full-attention family, and the feed-forward family, with the
//! chunk-terminal wait capturing the residual (route observations, final
//! logits).
//!
//! The method for a prefill optimization A/B: run this journey unchanged
//! before and after the change and compare the printed per-family totals and
//! shares. The per-family numbers are relative to this machine's graphics
//! processor, so compare within the same hardware.

use serde_json::{Value, json};
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, launch_resident_rest_server_with_prefill_attribution,
    prefill_attribution_warmup_prompt, resident_model_id, romeo_and_juliet_long_context_prompt,
    romeo_and_juliet_prompt, stop_resident_rest_server,
};

const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 16;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident 35B sparse-MoE model and attributes prefill graphics-processor time by op family"]
async fn should_attribute_resident_sparse_moe_prefill_graphics_processor_time_by_family() {
    timeout(
        JOURNEY_TIMEOUT,
        run_prefill_attribution_journey(romeo_and_juliet_prompt(), "short"),
    )
    .await
    .expect("the Ornith-35B prefill attribution journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident 35B sparse-MoE model and attributes long-context prefill graphics-processor time by op family"]
async fn should_attribute_resident_sparse_moe_long_context_prefill_graphics_processor_time_by_family()
 {
    timeout(
        JOURNEY_TIMEOUT,
        run_prefill_attribution_journey(romeo_and_juliet_long_context_prompt(), "long-context"),
    )
    .await
    .expect("the Ornith-35B long-context attribution journey must finish within 115 seconds");
}

async fn run_prefill_attribution_journey(prompt: String, context_label: &str) {
    let model_id = resident_model_id();
    let (isolated_home, rest_server) = launch_resident_rest_server_with_prefill_attribution().await;
    let server_address = rest_server.server_address;

    eprintln!("[ornith-35b-attribution] phase=jit-warmup model={model_id}");
    let warmup_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": prefill_attribution_warmup_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 8,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&warmup_response);

    eprintln!(
        "[ornith-35b-attribution] phase=measured-fresh-prefill context={context_label} model={model_id}"
    );
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": prompt,
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": MEASURED_MAXIMUM_OUTPUT_TOKENS,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);

    // The worker writes attribution records inside the isolated home, which it
    // owns until shutdown. Read them while the server is still up.
    let attribution_log_path = isolated_home
        .path()
        .join(".astronomical-dev/logs/performance-attribution.jsonl");
    let attribution_text = std::fs::read_to_string(&attribution_log_path)
        .expect("the attribution log must exist while the server is running");
    let measured_report = last_generation_report(&attribution_text)
        .expect("the measured completion must produce a generation attribution report");

    let prompt_token_count = counter_amount(&measured_report, "prompt_token_count")
        .expect("the measured report must count prompt tokens");
    let prefill_chunk_count = counter_amount(&measured_report, "prefill_chunk_count")
        .expect("the measured report must count prefill chunks");
    let prefill_advance_span = operation_total(&measured_report, "prompt_prefill_advance_span")
        .expect("the measured report must span the prompt prefill advance");
    let linear_attention = operation_total(
        &measured_report,
        "prefill_linear_attention_graphics_processor_completion_wait",
    )
    .expect("the measured report must attribute linear-attention prefill GPU waits");
    let full_attention = operation_total(
        &measured_report,
        "prefill_full_attention_graphics_processor_completion_wait",
    )
    .expect("the measured report must attribute full-attention prefill GPU waits");
    let feed_forward = operation_total(
        &measured_report,
        "prefill_feed_forward_graphics_processor_completion_wait",
    )
    .expect("the measured report must attribute feed-forward prefill GPU waits");
    let terminal_wait = operation_total(
        &measured_report,
        "prefill_state_graphics_processor_completion_wait",
    )
    .expect("the measured report must attribute the chunk-terminal prefill GPU wait");

    let prefill_span_millis = prefill_advance_span.total_millis;
    let prefill_tokens_per_second = prompt_token_count as f64 / (prefill_span_millis / 1000.0);
    eprintln!(
        "[ornith-35b-attribution] BASELINE model={model_id} prompt_tokens={prompt_token_count} prefill_chunks={prefill_chunk_count} prefill_span_ms={prefill_span_millis:.1} prefill_tok_per_second={prefill_tokens_per_second:.2}"
    );
    for (family_name, family_total) in [
        ("linear_attention", &linear_attention),
        ("full_attention", &full_attention),
        ("feed_forward", &feed_forward),
        ("terminal_wait", &terminal_wait),
    ] {
        eprintln!(
            "[ornith-35b-attribution] family={family_name} occurrences={} total_ms={:.1} share_of_prefill={:.1}%",
            family_total.occurrence_count,
            family_total.total_millis,
            100.0 * family_total.total_millis / prefill_span_millis,
        );
    }

    // Linear-attention sub-op attribution: where the GDN family's GPU time
    // lives inside one layer call (projections, convolution, normalization and
    // gates, recurrence kernel, epilogue). The family wait itself already
    // forces the epilogue output, so the five sections should sum to the
    // family total.
    let linear_attention_sections = [
        (
            "projections",
            "prefill_linear_attention_projections_graphics_processor_completion_wait",
        ),
        (
            "convolution",
            "prefill_linear_attention_convolution_graphics_processor_completion_wait",
        ),
        (
            "normalization",
            "prefill_linear_attention_normalization_graphics_processor_completion_wait",
        ),
        (
            "recurrence",
            "prefill_linear_attention_recurrence_graphics_processor_completion_wait",
        ),
        (
            "epilogue",
            "prefill_linear_attention_epilogue_graphics_processor_completion_wait",
        ),
    ];
    let measured_sections: Vec<(&str, OperationTotal)> = linear_attention_sections
        .iter()
        .filter_map(|(section_name, operation_name)| {
            operation_total(&measured_report, operation_name)
                .map(|section_total| (*section_name, section_total))
        })
        .collect();
    if !measured_sections.is_empty() {
        let attributed_sections_millis: f64 = measured_sections
            .iter()
            .map(|(_, section_total)| section_total.total_millis)
            .sum();
        for (section_name, section_total) in &measured_sections {
            eprintln!(
                "[ornith-35b-attribution] linear_attention_section={section_name} occurrences={} total_ms={:.1} share_of_linear_attention={:.1}%",
                section_total.occurrence_count,
                section_total.total_millis,
                100.0 * section_total.total_millis / linear_attention.total_millis,
            );
        }
        eprintln!(
            "[ornith-35b-attribution] linear_attention_sections_sum_ms={attributed_sections_millis:.1} family_total_ms={:.1}",
            linear_attention.total_millis,
        );
    }

    assert!(
        prompt_token_count > 1000,
        "the measured completion must process the full Romeo and Juliet prompt: {prompt_token_count} tokens"
    );
    assert!(
        prefill_chunk_count > 1,
        "the measured prefill must run as multiple chunks: {prefill_chunk_count} chunks"
    );
    assert!(
        prefill_span_millis > 0.0,
        "the prompt prefill advance span must be positive: {prefill_span_millis:.1} ms"
    );
    assert!(
        linear_attention.occurrence_count > 0 && linear_attention.total_millis > 0.0,
        "the linear-attention family must contribute prefill GPU time: {linear_attention:?}"
    );
    assert!(
        full_attention.occurrence_count > 0 && full_attention.total_millis > 0.0,
        "the full-attention family must contribute prefill GPU time: {full_attention:?}"
    );
    assert!(
        feed_forward.occurrence_count > 0 && feed_forward.total_millis > 0.0,
        "the feed-forward family must contribute prefill GPU time: {feed_forward:?}"
    );
    assert_eq!(
        feed_forward.occurrence_count,
        linear_attention.occurrence_count + full_attention.occurrence_count,
        "every decoder layer contributes exactly one attention-family wait and one feed-forward wait"
    );
    let attributed_graphics_millis = linear_attention.total_millis
        + full_attention.total_millis
        + feed_forward.total_millis
        + terminal_wait.total_millis;
    assert!(
        attributed_graphics_millis > 0.5 * prefill_span_millis,
        "prefill must be majority graphics-processor time: attributed {attributed_graphics_millis:.1} ms of {prefill_span_millis:.1} ms"
    );

    // Every attributed operation, largest first: this surfaces costs the
    // family boundaries do not own (routing, paging, graph construction) and
    // is how a context-growing anomaly gets named without a new journey.
    let mut all_operations: Vec<(String, u64, f64)> = measured_report["operations"]
        .as_array()
        .map(|operations| {
            operations
                .iter()
                .filter_map(|operation| {
                    let name = operation["operation"].as_str()?.to_owned();
                    let occurrence_count = operation["occurrence_count"].as_u64()?;
                    let total_millis =
                        operation["total_elapsed_nanoseconds"].as_u64()? as f64 / 1e6;
                    Some((name, occurrence_count, total_millis))
                })
                .collect()
        })
        .unwrap_or_default();
    all_operations.sort_by(|left, right| right.2.total_cmp(&left.2));
    for (operation_name, occurrence_count, total_millis) in all_operations.iter().take(15) {
        eprintln!(
            "[ornith-35b-attribution] operation={operation_name} occurrences={occurrence_count} total_ms={total_millis:.1} share_of_prefill={:.1}%",
            100.0 * total_millis / prefill_span_millis
        );
    }

    stop_resident_rest_server(rest_server).await;
}

/// One attributed operation aggregate from a report: how many times the
/// operation ran and the graphics/host time it owned in total.
#[derive(Debug)]
struct OperationTotal {
    occurrence_count: u64,
    total_millis: f64,
}

fn operation_total(report: &Value, operation_name: &str) -> Option<OperationTotal> {
    report
        .get("operations")?
        .as_array()?
        .iter()
        .find(|operation| operation["operation"] == operation_name)
        .map(|operation| OperationTotal {
            occurrence_count: operation["occurrence_count"].as_u64().unwrap_or(0),
            total_millis: operation["total_elapsed_nanoseconds"].as_u64().unwrap_or(0) as f64 / 1e6,
        })
}

fn counter_amount(report: &Value, counter_name: &str) -> Option<u64> {
    report
        .get("counters")?
        .as_array()?
        .iter()
        .find(|counter| counter["counter"] == counter_name)
        .and_then(|counter| counter["amount"].as_u64())
}

/// The latest generation report, which is the measured completion: the warmup
/// is the only earlier generation and its report precedes it in the log.
fn last_generation_report(attribution_text: &str) -> Option<Value> {
    attribution_text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["report_kind"] == "generation")
        .last()
}
