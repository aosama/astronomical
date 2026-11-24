use super::*;
use serde_json::json;

fn complete_status() -> Value {
    json!({
        "mlx_memory_ceiling_bytes": MEMORY_CEILING_BYTES,
        "expert_memory_mode": "resident",
        "expert_residency": {
            "total_layer_count": SPARSE_LAYER_COUNT,
            "resident_expert_count": SPARSE_LAYER_COUNT * EXPERT_CAPACITY,
            "resident_expert_payload_bytes": COMPLETE_EXPERT_PAYLOAD_BYTES
        },
        "mlx_memory_snapshot": {
            "active_memory_bytes": MEMORY_CEILING_BYTES,
            "expert_payload_bytes": COMPLETE_EXPERT_PAYLOAD_BYTES,
            "memory_ceiling_utilization": {
                "owner_overrun_bytes": 0,
                "unseated_expert_entitlement_bytes": 800_000_000
            }
        }
    })
}

fn generation_report() -> Value {
    json!({
        "report_kind": "generation",
        "counters": [
            {"counter": "prefill_chunk_count", "amount": 5},
            {"counter": "mandatory_prefill_expert_source_payload_bytes", "amount": 1},
            {"counter": "generated_token_count", "amount": 128}
        ]
    })
}

#[test]
fn should_accept_complete_residency_without_confusing_artifact_surplus_with_missing_experts() {
    assert_eq!(validate_decode_residency(&complete_status()), Ok(()));
    let mut pager_backed_status = complete_status();
    pager_backed_status["expert_memory_mode"] = json!("hybrid");
    assert_eq!(validate_decode_residency(&pager_backed_status), Ok(()));
    assert_eq!(
        validate_generation_attribution(&generation_report()),
        Ok(())
    );
}

#[test]
fn should_reject_the_partial_residency_and_unused_entitlement_observed_in_the_regression() {
    let mut status = complete_status();
    status["expert_memory_mode"] = json!("hybrid");
    status["expert_residency"]["resident_expert_count"] = json!(SPARSE_LAYER_COUNT * 64);
    status["expert_residency"]["resident_expert_payload_bytes"] =
        json!(SPARSE_LAYER_COUNT * 64 * EXPERT_PAYLOAD_BYTES);
    assert!(validate_decode_residency(&status).is_err());
}

#[test]
fn should_reject_one_missing_expert_or_a_clamped_ceiling() {
    let mut status = complete_status();
    status["expert_residency"]["resident_expert_count"] =
        json!(SPARSE_LAYER_COUNT * EXPERT_CAPACITY - 1);
    assert!(validate_decode_residency(&status).is_err());
    status = complete_status();
    status["mlx_memory_ceiling_bytes"] = json!(36 * (1_u64 << 30));
    assert!(validate_decode_residency(&status).is_err());
    status = complete_status();
    status["expert_memory_mode"] = json!("paged");
    assert!(validate_decode_residency(&status).is_err());
}

#[test]
fn should_reject_one_byte_above_the_ceiling_or_an_owner_overrun() {
    let mut status = complete_status();
    status["mlx_memory_snapshot"]["active_memory_bytes"] = json!(MEMORY_CEILING_BYTES + 1);
    assert!(validate_decode_residency(&status).is_err());
    status = complete_status();
    status["mlx_memory_snapshot"]["memory_ceiling_utilization"]["owner_overrun_bytes"] = json!(1);
    assert!(validate_decode_residency(&status).is_err());
}

#[test]
fn should_reject_decode_storage_reads_and_missing_or_single_chunk_evidence() {
    let mut report = generation_report();
    report["counters"].as_array_mut().unwrap().push(json!({
        "counter": "mandatory_decode_expert_source_payload_bytes", "amount": 1
    }));
    assert!(validate_generation_attribution(&report).is_err());
    report = generation_report();
    report["counters"][0]["amount"] = json!(1);
    assert!(validate_generation_attribution(&report).is_err());
    report = generation_report();
    report["counters"][1]["amount"] = json!(0);
    assert!(validate_generation_attribution(&report).is_err());
    assert!(validate_generation_attribution(&json!({"report_kind": "generation"})).is_err());
}

#[test]
fn should_skip_the_resident_admission_attempt_before_the_paged_generation_report() {
    let logging_directory = tempfile::tempdir().unwrap();
    let mut completed_report = generation_report();
    completed_report["request_id"] = json!(2);
    let admission_report = json!({
        "report_kind": "generation", "request_id": 2, "counters": []
    });
    std::fs::write(
        logging_directory
            .path()
            .join("performance-attribution.jsonl"),
        format!("{admission_report}\n{completed_report}\n"),
    )
    .unwrap();
    assert_eq!(
        read_generation_attribution_report(logging_directory.path(), 2),
        Ok(completed_report)
    );
    assert!(read_generation_attribution_report(logging_directory.path(), 3).is_err());
}
