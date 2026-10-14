//! Evidence preservation and status observation for the decode expert reuse
//! journey (issue #510): durable utilization evidence plus the status polling
//! helpers the journey reads the user-visible memory numbers through.

use std::{fs, path::Path};

use serde_json::Value;

pub(super) fn preserve_memory_utilization_evidence(
    isolated_worker_home: &Path,
    final_status: &Value,
    counters: &[(&str, u64)],
) {
    let unix_millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|unix_time| unix_time.as_millis())
        .unwrap_or_default();
    let evidence_directory = std::env::current_dir()
        .expect("the journey should resolve its working directory")
        .join("target/acceptance-evidence/ssd-paging-decode-expert-reuse")
        .join(unix_millis.to_string());
    fs::create_dir_all(&evidence_directory)
        .expect("the acceptance evidence directory should be created");
    let mut counter_document = serde_json::Map::new();
    for (counter_identifier, counter_amount) in counters {
        counter_document.insert(
            (*counter_identifier).to_owned(),
            serde_json::json!(counter_amount),
        );
    }
    let evidence_document = serde_json::json!({
        "attribution_counters": counter_document,
        "status_memory_ceiling_utilization": final_status["mlx_memory_snapshot"]["memory_ceiling_utilization"],
        "status_mlx_memory_snapshot": final_status["mlx_memory_snapshot"],
    });
    fs::write(
        evidence_directory.join("memory-utilization.json"),
        serde_json::to_string_pretty(&evidence_document).expect("evidence should serialize"),
    )
    .expect("the utilization evidence should be written");
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let Ok(logging_entries) = fs::read_dir(&logging_directory) else {
        return;
    };
    for logging_entry in logging_entries.flatten() {
        let source_path = logging_entry.path();
        if source_path.is_file()
            && let Some(source_name) = source_path.file_name()
        {
            let _ = fs::copy(&source_path, evidence_directory.join(source_name));
        }
    }
}

pub(super) fn generation_attribution_counter(
    isolated_worker_home: &Path,
    counter_identifier: &str,
) -> u64 {
    crate::support::attribution_log::generation_attribution_counter(
        isolated_worker_home,
        counter_identifier,
    )
}

pub(super) fn generation_attribution_report_count(isolated_worker_home: &Path) -> usize {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .count()
}

pub(super) fn decode_streamed_layer_indices(
    isolated_worker_home: &Path,
) -> std::collections::BTreeSet<usize> {
    isolated_worker_log_lines(isolated_worker_home)
        .into_iter()
        .filter(|log_line| {
            log_line.contains("Rust expert layer streaming completed")
                && !log_line.contains("streamed_expert_count=256")
        })
        .filter_map(|log_line| {
            log_line
                .split_whitespace()
                .find_map(|field| field.strip_prefix("layer_index="))
                .and_then(|layer_index| layer_index.parse::<usize>().ok())
        })
        .collect()
}

pub(super) fn generation_expert_source_read_bytes(isolated_worker_home: &Path) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the paging acceptance journey should write performance attribution");
    attribution_log
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .filter_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .map(|counters| counters.to_owned())
        })
        .flatten()
        .filter(|counter_report| counter_report["counter"] == "positional_file_read_byte_count")
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

pub(super) fn isolated_worker_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let logging_entries = fs::read_dir(logging_directory)
        .expect("the paging acceptance journey should create its logging directory");
    let mut log_lines = Vec::new();
    for logging_entry in logging_entries {
        let log_path = logging_entry
            .expect("the isolated log entry should be readable")
            .path();
        if log_path.is_file() {
            let log_content = fs::read_to_string(&log_path).unwrap_or_else(|log_read_error| {
                panic!(
                    "{} should be readable: {log_read_error}",
                    log_path.display()
                )
            });
            log_lines.extend(log_content.lines().map(str::to_owned));
        }
    }
    log_lines
}

pub(super) fn log_status_progress(status_document: &Value) {
    let phase = status_document["progress"]["phase"]
        .as_str()
        .unwrap_or("idle");
    let processed_tokens = status_document["progress"]["processed_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total_tokens = status_document["progress"]["total_tokens"]
        .as_u64()
        .unwrap_or(0);
    let elapsed_millis = status_document["progress"]["elapsed_ms"]
        .as_u64()
        .unwrap_or(0);
    let observed_tokens_per_second = if elapsed_millis == 0 {
        0.0
    } else {
        processed_tokens as f64 * 1_000.0 / elapsed_millis as f64
    };
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "[ssd-paging-decode-expert-reuse] status=progress phase={phase} processed_tokens={processed_tokens} total_tokens={total_tokens} elapsed_seconds={:.3} observed_tokens_per_second={observed_tokens_per_second:.2} expert_payload_bytes={expert_payload_bytes}",
        elapsed_millis as f64 / 1_000.0,
    );
}

pub(super) fn generation_attribution_counter_identifiers(
    isolated_worker_home: &Path,
) -> std::collections::BTreeSet<String> {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .flat_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|counter_report| counter_report["counter"].as_str().map(str::to_string))
        .collect()
}

/// The predictor serving surface is retired (#594): no counter it produced may
/// appear in generation attribution, and worker status must not carry a
/// predictor object. Route-observation capture (#536) and the previous-token
/// prefetch baseline (#537) remain the surviving evidence paths.
pub(super) fn assert_predictor_surface_absent(
    final_status: &Value,
    attribution_counter_identifiers: &std::collections::BTreeSet<String>,
) {
    let retired_counter_identifiers = [
        "expert_route_predictor_train_step_count",
        "expert_route_predictor_top_k_hit_count",
        "expert_route_predictor_evaluated_expert_count",
        "expert_route_predictor_train_slice_nanoseconds",
        "expert_route_predictor_cpu_predict_nanoseconds",
        "predictor_retention_hint_requested_count",
        "predictor_retention_hint_accepted_count",
        "predictor_prefetch_issue_count",
        "predictor_prefetch_byte_count",
    ];
    for retired_counter_identifier in retired_counter_identifiers {
        assert!(
            !attribution_counter_identifiers.contains(retired_counter_identifier),
            "retired predictor counter {retired_counter_identifier} must not be recorded after issue #594"
        );
    }
    assert!(
        final_status.get("predictor").is_none(),
        "the retired predictor surface must not publish a status object (#594)"
    );
}

pub(super) fn record_expert_payload_increase(
    status_document: &Value,
    retained_expert_payload_bytes: &mut Vec<u64>,
) {
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    let largest_recorded_expert_payload_bytes = retained_expert_payload_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if expert_payload_bytes > largest_recorded_expert_payload_bytes {
        retained_expert_payload_bytes.push(expert_payload_bytes);
        eprintln!(
            "[ssd-paging-decode-expert-reuse] status=progress processed_tokens={} expert_payload_bytes={expert_payload_bytes}",
            status_document["progress"]["processed_tokens"]
        );
    }
}

/// Every generation-attribution counter the reuse journey asserts on, fetched
/// once so the asserts and the success evidence print the same numbers.
pub(super) struct DecodeExpertReuseCounters {
    pub(super) retained_route_assignment_hit_count: u64,
    pub(super) hot_expert_partial_route_hit_count: u64,
    pub(super) hot_expert_warm_insert_count: u64,
    pub(super) hot_expert_route_fully_covered_count: u64,
    pub(super) hot_expert_route_partially_covered_count: u64,
    pub(super) hot_expert_route_fully_missed_count: u64,
    pub(super) hot_expert_route_retained_assignment_count: u64,
    pub(super) hot_expert_route_missing_assignment_count: u64,
    pub(super) hot_expert_mixed_route_count: u64,
    pub(super) route_observation_stored_record_count: u64,
    pub(super) route_observation_captured_layer_count: u64,
    pub(super) route_observation_evicted_record_count: u64,
    pub(super) previous_token_prefetch_issue_count: u64,
    pub(super) previous_token_prefetch_hit_count: u64,
    pub(super) previous_token_prefetch_miss_count: u64,
    pub(super) previous_token_prefetch_byte_count: u64,
    pub(super) previous_token_prefetch_capacity_drop_count: u64,
    pub(super) memory_ceiling_utilization_ceiling_bytes: u64,
    pub(super) memory_ceiling_utilization_active_bytes: u64,
    pub(super) memory_ceiling_utilization_unused_headroom_bytes: u64,
    pub(super) memory_ceiling_utilization_reserved_model_core_slack_bytes: u64,
    pub(super) memory_ceiling_utilization_reserved_context_growth_bytes: u64,
    pub(super) memory_ceiling_utilization_reserved_activation_and_workspace_bytes: u64,
    pub(super) memory_ceiling_utilization_unseated_expert_entitlement_bytes: u64,
    pub(super) memory_ceiling_utilization_unexplained_headroom_bytes: u64,
    pub(super) memory_ceiling_utilization_owner_overrun_bytes: u64,
}

/// Fetches the generation-attribution counters and asserts the routed-reuse
/// behavior they prove: the hot-expert warm table is live (#372), route
/// coverage is classified per token and mixed routes serve from retained RAM
/// (#373), true routes are observed (#536), the predictor surface stays
/// retired (#537, #594), and the ceiling-utilization identity closes with no
/// unexplained residual (#507).
pub(super) fn assert_decode_expert_reuse_attribution(
    attribution_home: &Path,
    final_status: &Value,
) -> DecodeExpertReuseCounters {
    let fetch_counter = |counter_identifier: &str| {
        generation_attribution_counter(attribution_home, counter_identifier)
    };
    let counters = DecodeExpertReuseCounters {
        retained_route_assignment_hit_count: fetch_counter("retained_route_assignment_hit_count"),
        hot_expert_partial_route_hit_count: fetch_counter("hot_expert_partial_route_hit_count"),
        hot_expert_warm_insert_count: fetch_counter("hot_expert_warm_insert_count"),
        hot_expert_route_fully_covered_count: fetch_counter("hot_expert_route_fully_covered_count"),
        hot_expert_route_partially_covered_count: fetch_counter(
            "hot_expert_route_partially_covered_count",
        ),
        hot_expert_route_fully_missed_count: fetch_counter("hot_expert_route_fully_missed_count"),
        hot_expert_route_retained_assignment_count: fetch_counter(
            "hot_expert_route_retained_assignment_count",
        ),
        hot_expert_route_missing_assignment_count: fetch_counter(
            "hot_expert_route_missing_assignment_count",
        ),
        hot_expert_mixed_route_count: fetch_counter("hot_expert_mixed_route_count"),
        route_observation_stored_record_count: fetch_counter(
            "route_observation_stored_record_count",
        ),
        route_observation_captured_layer_count: fetch_counter(
            "route_observation_captured_layer_count",
        ),
        route_observation_evicted_record_count: fetch_counter(
            "route_observation_evicted_record_count",
        ),
        previous_token_prefetch_issue_count: fetch_counter("previous_token_prefetch_issue_count"),
        previous_token_prefetch_hit_count: fetch_counter("previous_token_prefetch_hit_count"),
        previous_token_prefetch_miss_count: fetch_counter("previous_token_prefetch_miss_count"),
        previous_token_prefetch_byte_count: fetch_counter("previous_token_prefetch_byte_count"),
        previous_token_prefetch_capacity_drop_count: fetch_counter(
            "previous_token_prefetch_capacity_drop_count",
        ),
        memory_ceiling_utilization_ceiling_bytes: fetch_counter(
            "memory_ceiling_utilization_ceiling_bytes",
        ),
        memory_ceiling_utilization_active_bytes: fetch_counter(
            "memory_ceiling_utilization_active_bytes",
        ),
        memory_ceiling_utilization_unused_headroom_bytes: fetch_counter(
            "memory_ceiling_utilization_unused_headroom_bytes",
        ),
        memory_ceiling_utilization_reserved_model_core_slack_bytes: fetch_counter(
            "memory_ceiling_utilization_reserved_model_core_slack_bytes",
        ),
        memory_ceiling_utilization_reserved_context_growth_bytes: fetch_counter(
            "memory_ceiling_utilization_reserved_context_growth_bytes",
        ),
        memory_ceiling_utilization_reserved_activation_and_workspace_bytes: fetch_counter(
            "memory_ceiling_utilization_reserved_activation_and_workspace_bytes",
        ),
        memory_ceiling_utilization_unseated_expert_entitlement_bytes: fetch_counter(
            "memory_ceiling_utilization_unseated_expert_entitlement_bytes",
        ),
        memory_ceiling_utilization_unexplained_headroom_bytes: fetch_counter(
            "memory_ceiling_utilization_unexplained_headroom_bytes",
        ),
        memory_ceiling_utilization_owner_overrun_bytes: fetch_counter(
            "memory_ceiling_utilization_owner_overrun_bytes",
        ),
    };

    // The routed-reuse branch is live (#372): unseated layers must warm routed
    // experts during decode and serve later tokens from the hot set instead of
    // re-reading storage every token.
    assert!(
        counters.hot_expert_partial_route_hit_count > 0
            && counters.hot_expert_warm_insert_count > 0,
        "decode must warm routed experts and reuse them; hot_expert_partial_route_hit_count={} hot_expert_warm_insert_count={}",
        counters.hot_expert_partial_route_hit_count,
        counters.hot_expert_warm_insert_count,
    );

    // Route-coverage classification (#373): the all-or-nothing warm-table rule
    // serves a token from RAM only when every routed expert is warm, so every
    // decode token must be classified and mixed routes must be measured, with
    // the majority of assignments served from retained RAM.
    let classified_token_count = counters.hot_expert_route_fully_covered_count
        + counters.hot_expert_route_partially_covered_count
        + counters.hot_expert_route_fully_missed_count;
    assert!(
        classified_token_count > 0,
        "decode must classify route coverage per token; classified_token_count={classified_token_count}"
    );
    assert!(
        counters.hot_expert_mixed_route_count > 0,
        "partially covered tokens must be served by the mixed route instead of reading every routed expert from storage; hot_expert_mixed_route_count={} partially_covered={}",
        counters.hot_expert_mixed_route_count,
        counters.hot_expert_route_partially_covered_count,
    );
    assert!(
        counters.hot_expert_route_retained_assignment_count
            > counters.hot_expert_route_missing_assignment_count,
        "mixed serving must serve the majority of routed assignments from retained RAM; retained={} missing={}",
        counters.hot_expert_route_retained_assignment_count,
        counters.hot_expert_route_missing_assignment_count,
    );

    // True-route observation history (#536): paged decode with attribution
    // enabled must capture real layer routes, not just table statistics.
    assert!(
        counters.route_observation_stored_record_count > 0,
        "paged decode with attribution enabled must capture at least one true route observation; stored={}",
        counters.route_observation_stored_record_count,
    );
    assert!(
        counters.route_observation_captured_layer_count > 0,
        "captured observations must include at least one sparse layer route; captured_layers={}",
        counters.route_observation_captured_layer_count,
    );

    // Previous-token prefetch survives; the predictor surface stays retired
    // (#537, #594).
    assert_predictor_surface_absent(
        final_status,
        &generation_attribution_counter_identifiers(attribution_home),
    );

    // The unused ceiling has a named owner (#507): every unused byte must be
    // attributed and the utilization identity must close with no residual.
    assert!(
        counters.memory_ceiling_utilization_unused_headroom_bytes > 0,
        "decode must observe unused headroom to explain; memory_unused_headroom_bytes={}",
        counters.memory_ceiling_utilization_unused_headroom_bytes,
    );
    let named_headroom_bytes = counters.memory_ceiling_utilization_reserved_model_core_slack_bytes
        + counters.memory_ceiling_utilization_reserved_context_growth_bytes
        + counters.memory_ceiling_utilization_reserved_activation_and_workspace_bytes
        + counters.memory_ceiling_utilization_unseated_expert_entitlement_bytes;
    assert!(
        counters.memory_ceiling_utilization_unexplained_headroom_bytes * 100
            <= counters.memory_ceiling_utilization_unused_headroom_bytes,
        "every unused byte must have a named owner or surface as a small residual; unused={} named={named_headroom_bytes} unexplained={}",
        counters.memory_ceiling_utilization_unused_headroom_bytes,
        counters.memory_ceiling_utilization_unexplained_headroom_bytes,
    );
    assert_eq!(
        counters.memory_ceiling_utilization_unexplained_headroom_bytes,
        0,
        "the utilization identity must close with no residual; unused={} named={named_headroom_bytes} unexplained={}",
        counters.memory_ceiling_utilization_unused_headroom_bytes,
        counters.memory_ceiling_utilization_unexplained_headroom_bytes,
    );
    counters
}
