use super::*;

#[test]
fn should_parse_the_measured_generation_cache_evidence() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    let logging_directory =
        tempfile::tempdir().expect("the worker logging directory should be created");
    let warmup_report = serde_json::json!({
        "report_kind": "generation",
        "request_id": 1,
        "counters": []
    });
    let measured_report = serde_json::json!({
        "report_kind": "generation",
        "request_id": 42,
        "decode_token_forward_latency_p95_nanoseconds": 21_000_000,
        "decode_token_forward_latency_histogram": [
            {"upper_bound_nanoseconds": 20_000_000, "sample_count": 9},
            {"upper_bound_nanoseconds": 21_000_000, "sample_count": 1}
        ],
        "operations": [
            {
                "operation": "decode_advance_span",
                "occurrence_count": 2,
                "total_elapsed_nanoseconds": 80,
                "minimum_elapsed_nanoseconds": 30,
                "maximum_elapsed_nanoseconds": 50
            },
            {
                "operation": "decode_expert_route_host_materialization",
                "occurrence_count": 4,
                "total_elapsed_nanoseconds": 60,
                "minimum_elapsed_nanoseconds": 10,
                "maximum_elapsed_nanoseconds": 20
            },
            {
                "operation": "decode_expert_route_contiguous_array_construction",
                "occurrence_count": 4,
                "total_elapsed_nanoseconds": 16,
                "minimum_elapsed_nanoseconds": 2,
                "maximum_elapsed_nanoseconds": 6
            },
            {
                "operation": "decode_expert_route_array_evaluation",
                "occurrence_count": 4,
                "total_elapsed_nanoseconds": 36,
                "minimum_elapsed_nanoseconds": 6,
                "maximum_elapsed_nanoseconds": 12
            },
            {
                "operation": "decode_expert_route_host_copy",
                "occurrence_count": 4,
                "total_elapsed_nanoseconds": 8,
                "minimum_elapsed_nanoseconds": 1,
                "maximum_elapsed_nanoseconds": 3
            },
            {
                "operation": "rust_expert_streaming_layer_preparation",
                "occurrence_count": 4,
                "total_elapsed_nanoseconds": 40,
                "minimum_elapsed_nanoseconds": 5,
                "maximum_elapsed_nanoseconds": 15
            }
        ],
        "expert_streaming_source_summaries": [
            {"phase": "prefill", "payload_byte_count": 900},
            {"phase": "decode", "payload_byte_count": 321}
        ],
        "counters": [
            {"counter": "mandatory_decode_expert_source_payload_bytes", "amount": 321},
            {"counter": "mandatory_prefill_expert_source_payload_bytes", "amount": 900},
            {"counter": "retained_route_assignment_hit_count", "amount": 8},
            {"counter": "retained_route_assignment_miss_count", "amount": 2},
            {"counter": "hot_expert_route_fully_covered_count", "amount": 1},
            {"counter": "hot_expert_route_partially_covered_count", "amount": 2},
            {"counter": "hot_expert_route_fully_missed_count", "amount": 3},
            {"counter": "decode_expert_cache_slot_eviction_count", "amount": 4},
            {"counter": "positional_file_read_call_count", "amount": 8},
            {"counter": "positional_file_read_byte_count", "amount": 1_024},
            {"counter": "positional_file_read_elapsed_nanoseconds", "amount": 9_000},
            {"counter": "positional_file_read_maximum_elapsed_nanoseconds", "amount": 3_000},
            {"counter": "positional_file_read_maximum_concurrent_count", "amount": 2}
        ],
        "process_physical_disk_read_bytes": 2_048,
        "mlx_peak_memory_bytes": 3_072
    });
    fs::write(
        logging_directory
            .path()
            .join("performance-attribution.jsonl"),
        format!("{warmup_report}\n{measured_report}\n"),
    )
    .expect("the attribution reports should be written");

    let diagnostic_evidence = read_decode_cache_diagnostic_evidence(logging_directory.path(), 42)
        .expect("the measured request report should be parsed");

    assert_eq!(
        diagnostic_evidence.decode_token_forward_latency_p95_nanoseconds,
        Some(21_000_000)
    );
    assert_eq!(
        diagnostic_evidence.decode_token_forward_latency_sample_count,
        Some(10)
    );
    assert_eq!(
        diagnostic_evidence
            .decode_advance_span
            .as_ref()
            .map(|operation_evidence| operation_evidence.occurrence_count),
        Some(2)
    );
    assert_eq!(
        diagnostic_evidence
            .decode_expert_route_host_materialization
            .as_ref()
            .map(|operation_evidence| operation_evidence.total_elapsed_nanoseconds),
        Some(60)
    );
    assert_eq!(
        diagnostic_evidence
            .decode_expert_route_contiguous_array_construction
            .as_ref()
            .map(|operation_evidence| operation_evidence.total_elapsed_nanoseconds),
        Some(16)
    );
    assert_eq!(
        diagnostic_evidence
            .decode_expert_route_array_evaluation
            .as_ref()
            .map(|operation_evidence| operation_evidence.total_elapsed_nanoseconds),
        Some(36)
    );
    assert_eq!(
        diagnostic_evidence
            .decode_expert_route_host_copy
            .as_ref()
            .map(|operation_evidence| operation_evidence.total_elapsed_nanoseconds),
        Some(8)
    );
    assert_eq!(
        diagnostic_evidence
            .rust_expert_streaming_layer_preparation
            .as_ref()
            .map(|operation_evidence| operation_evidence.maximum_elapsed_nanoseconds),
        Some(15)
    );
    assert_eq!(diagnostic_evidence.decode_expert_source_payload_bytes, 321);
    assert_eq!(
        diagnostic_evidence.decode_expert_source_summary_payload_bytes,
        321
    );
    assert_eq!(diagnostic_evidence.prefill_expert_source_payload_bytes, 900);
    assert_eq!(diagnostic_evidence.retained_route_assignment_hit_count, 8);
    assert_eq!(diagnostic_evidence.retained_route_assignment_miss_count, 2);
    assert_eq!(diagnostic_evidence.hot_expert_route_fully_covered_count, 1);
    assert_eq!(
        diagnostic_evidence.hot_expert_route_partially_covered_count,
        2
    );
    assert_eq!(diagnostic_evidence.hot_expert_route_fully_missed_count, 3);
    assert_eq!(
        diagnostic_evidence.decode_expert_cache_slot_eviction_count,
        4
    );
    assert_eq!(diagnostic_evidence.positional_file_read_call_count, 8);
    assert_eq!(diagnostic_evidence.positional_file_read_byte_count, 1_024);
    assert_eq!(
        diagnostic_evidence.positional_file_read_elapsed_nanoseconds,
        9_000
    );
    assert_eq!(
        diagnostic_evidence.positional_file_read_maximum_elapsed_nanoseconds,
        3_000
    );
    assert_eq!(
        diagnostic_evidence.positional_file_read_maximum_concurrent_count,
        2
    );
    assert_eq!(
        diagnostic_evidence.process_physical_disk_read_bytes,
        Some(2_048)
    );
    assert_eq!(diagnostic_evidence.mlx_peak_memory_bytes, Some(3_072));
}

#[test]
fn should_not_use_warmup_evidence_when_the_measured_report_is_missing() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    let logging_directory =
        tempfile::tempdir().expect("the worker logging directory should be created");
    let warmup_report = serde_json::json!({
        "report_kind": "generation",
        "request_id": 1,
        "counters": []
    });
    fs::write(
        logging_directory
            .path()
            .join("performance-attribution.jsonl"),
        format!("{warmup_report}\n"),
    )
    .expect("the warmup attribution report should be written");

    assert!(
        read_decode_cache_diagnostic_evidence(logging_directory.path(), 42).is_none(),
        "a warmup report must not stand in for missing measured evidence"
    );
}

#[test]
fn should_treat_omitted_zero_counters_as_zero() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");
    let logging_directory =
        tempfile::tempdir().expect("the worker logging directory should be created");
    let measured_report = serde_json::json!({
        "report_kind": "generation",
        "request_id": 42,
        "expert_streaming_source_summaries": [],
        "counters": []
    });
    fs::write(
        logging_directory
            .path()
            .join("performance-attribution.jsonl"),
        format!("{measured_report}\n"),
    )
    .expect("the measured attribution report should be written");

    let diagnostic_evidence = read_decode_cache_diagnostic_evidence(logging_directory.path(), 42)
        .expect("an empty valid report should parse");

    assert_eq!(diagnostic_evidence.decode_expert_source_payload_bytes, 0);
    assert_eq!(
        diagnostic_evidence.decode_expert_cache_slot_eviction_count,
        0
    );
    assert_eq!(
        diagnostic_evidence.decode_token_forward_latency_sample_count,
        None
    );
    assert!(diagnostic_evidence.decode_advance_span.is_none());
    assert!(
        diagnostic_evidence
            .decode_expert_route_host_materialization
            .is_none()
    );
}
