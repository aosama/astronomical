//! Shared byte-level oracle for the 37-gigabyte sparse-model regression cell.

use serde_json::Value;
use std::path::Path;

pub const MEMORY_CEILING_BYTES: u64 = 37 * (1 << 30);
pub const SPARSE_LAYER_COUNT: u64 = 40;
pub const EXPERT_CAPACITY: u64 = 256;
pub const EXPERT_PAYLOAD_BYTES: u64 = 3_342_336;
pub const COMPLETE_EXPERT_PAYLOAD_BYTES: u64 =
    SPARSE_LAYER_COUNT * EXPERT_CAPACITY * EXPERT_PAYLOAD_BYTES;

pub fn configuration_document(model_directory: &Path) -> Value {
    serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [model_directory],
            "maximum_mlx_memory_gb": 37
        },
        "prompt_cache": {"enabled": false},
        "diagnostics": {"performance_attribution_enabled": true, "log_level": "info"},
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2048
        }
    })
}

pub fn validate_decode_residency(status: &Value) -> Result<(), String> {
    for (field_path, expected_bytes) in [
        ("/mlx_memory_ceiling_bytes", MEMORY_CEILING_BYTES),
        ("/expert_residency/total_layer_count", SPARSE_LAYER_COUNT),
        (
            "/expert_residency/resident_expert_count",
            SPARSE_LAYER_COUNT * EXPERT_CAPACITY,
        ),
        (
            "/expert_residency/resident_expert_payload_bytes",
            COMPLETE_EXPERT_PAYLOAD_BYTES,
        ),
        (
            "/mlx_memory_snapshot/expert_payload_bytes",
            COMPLETE_EXPERT_PAYLOAD_BYTES,
        ),
    ] {
        let observed_bytes = status.pointer(field_path).and_then(Value::as_u64);
        if observed_bytes != Some(expected_bytes) {
            return Err(format!(
                "full-fit decode requires {field_path}={expected_bytes}, observed {observed_bytes:?}: {status}"
            ));
        }
    }
    // Pager-backed complete tables remain structurally Hybrid. The exact
    // expert count and payload, not the engine ownership label, prove full fit.
    if !matches!(
        status["expert_memory_mode"].as_str(),
        Some("resident" | "hybrid")
    ) {
        return Err(format!(
            "full-fit decode must report retained expert ownership: {status}"
        ));
    }
    let utilization = &status["mlx_memory_snapshot"]["memory_ceiling_utilization"];
    let active_memory_bytes = status["mlx_memory_snapshot"]["active_memory_bytes"]
        .as_u64()
        .ok_or_else(|| format!("decode must publish active memory: {status}"))?;
    if active_memory_bytes > MEMORY_CEILING_BYTES
        || utilization["owner_overrun_bytes"].as_u64() != Some(0)
    {
        return Err(format!(
            "complete decode residency exceeds its ceiling: {status}"
        ));
    }
    let _unseated_entitlement_bytes = utilization["unseated_expert_entitlement_bytes"]
        .as_u64()
        .ok_or_else(|| format!("decode must publish unseated expert entitlement: {status}"))?;
    // The budget can exceed the entire artifact. Only entitlement backed by
    // missing experts is recoverable; exact payload equality above proves none
    // remain, while the raw surplus remains visible in the captured status.
    Ok(())
}

pub fn attribution_counter(report: &Value, counter_name: &str) -> Result<u64, String> {
    let counters = report["counters"]
        .as_array()
        .ok_or_else(|| "the generation report must contain counter evidence".to_owned())?;
    counters
        .iter()
        .filter(|counter| counter["counter"] == counter_name)
        .try_fold(0_u64, |total_amount, counter| {
            let amount = counter["amount"]
                .as_u64()
                .ok_or_else(|| format!("invalid {counter_name} counter: {counter}"))?;
            total_amount
                .checked_add(amount)
                .ok_or_else(|| format!("{counter_name} counter overflowed"))
        })
}

pub fn validate_generation_attribution(report: &Value) -> Result<(), String> {
    if report["report_kind"] != "generation" {
        return Err("the residency oracle requires a generation report".to_owned());
    }
    for (counter_name, minimum_amount) in [
        ("prefill_chunk_count", 2),
        ("mandatory_prefill_expert_source_payload_bytes", 1),
        ("generated_token_count", 1),
    ] {
        if attribution_counter(report, counter_name)? < minimum_amount {
            return Err(format!(
                "the journey did not exercise {counter_name}: {report}"
            ));
        }
    }
    let decode_source_bytes =
        attribution_counter(report, "mandatory_decode_expert_source_payload_bytes")?;
    if decode_source_bytes != 0 {
        return Err(format!(
            "full-fit decode must not read expert storage after handoff; read {decode_source_bytes} bytes"
        ));
    }
    Ok(())
}

pub fn read_generation_attribution_report(
    logging_directory: &Path,
    request_id: u64,
) -> Result<Value, String> {
    let attribution_text =
        std::fs::read_to_string(logging_directory.join("performance-attribution.jsonl"))
            .map_err(|failure| format!("cannot read generation attribution: {failure}"))?;
    for line in attribution_text.lines() {
        let report: Value = serde_json::from_str(line)
            .map_err(|failure| format!("invalid attribution record: {failure}"))?;
        if report["report_kind"] == "generation"
            && report["request_id"].as_u64() == Some(request_id)
            && attribution_counter(&report, "generated_token_count")? > 0
        {
            return Ok(report);
        }
    }
    Err(format!(
        "no generation attribution for request {request_id}"
    ))
}

#[cfg(test)]
#[path = "sparse_decode_full_fit_tests.rs"]
mod tests;
