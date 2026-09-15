//! Controlled MTP on/off decode A/B for the resident Qwen3.6-35B-A3B OptiQ-4bit artifact.
//!
//! The journey launches the production worker twice against the same artifact —
//! once with the authored MTP acceleration block removed from the isolated home's
//! config, once with it present — and reads server-attributed throughput counters
//! from `/v1/status` after an identical Romeo and Juliet chat request. Both runs
//! use temperature 1 so the comparison reflects real serving behavior.

use std::time::Duration;

use serde_json::{Value, json};
use tokio::time::timeout;

use super::super::chat::openai_rest::{
    get_endpoint, launch_serving_rest_server_for_model, post_chat_completion,
    stop_serving_rest_server,
};
use crate::support::isolated_development_home_from_user_config;

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const OUTPUT_TOKEN_COUNT: u16 = 256;
const MTP_MODEL_ID: &str = "Qwen3.6-35B-A3B-OptiQ-4bit";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "measures resident MTP decode A/B against the real Qwen3.6 OptiQ-4bit artifact"]
async fn should_measure_mtp_decode_against_the_resident_qwen3_6_optiq_baseline() {
    timeout(
        JOURNEY_TIMEOUT,
        run_mtp_ab_journey("mtp-off", |isolated_home_config| {
            remove_mtp_acceleration(isolated_home_config);
        }),
    )
    .await
    .expect("the MTP-off baseline run must finish within 115 seconds");
    timeout(JOURNEY_TIMEOUT, run_mtp_ab_journey("mtp-on", |_| {}))
        .await
        .expect("the MTP-on measured run must finish within 115 seconds");
}

async fn run_mtp_ab_journey(
    phase_label: &str,
    adjust_isolated_home_config: impl Fn(&std::path::Path),
) {
    let model_directory = crate::support::configured_installed_model_directory_by_id(MTP_MODEL_ID);
    let isolated_development_home = isolated_development_home_from_user_config();
    adjust_isolated_home_config(isolated_development_home.path());
    // A persistent log directory keeps the worker's attribution records alive
    // after the journey ends, so MTP draft/accept counters can be inspected.
    let performance_log_directory = std::env::temp_dir().join("astronomical-mtp-ab-attribution");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the persistent MTP attribution log directory must be created");
    eprintln!("[mtp-ab] phase=launch phase={phase_label} model={MTP_MODEL_ID}");
    let rest_server = launch_serving_rest_server_for_model(
        MTP_MODEL_ID,
        model_directory,
        Some(isolated_development_home.path()),
        Some(&performance_log_directory),
    )
    .await;
    let server_address = rest_server.server_address;

    eprintln!("[mtp-ab] phase=generate phase={phase_label}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": MTP_MODEL_ID,
            "messages": [{
                "role": "user",
                "content": "Summarize the feud between the two households in Romeo and Juliet in your own words.",
            }],
            "stream": false,
            "temperature": 1,
            "max_tokens": OUTPUT_TOKEN_COUNT,
        })
        .to_string(),
    )
    .await;
    assert!(
        chat_response.contains("200 OK"),
        "the chat completion must succeed in phase {phase_label}: {chat_response}"
    );

    let status_document = read_status_document(server_address, phase_label).await;
    let serving_session = &status_document["serving_session"];
    let prefill_tokens_per_second = serving_session["average_prefill_tok_per_second"]
        .as_f64()
        .unwrap_or_else(|| panic!("status must report prefill throughput: {status_document}"));
    let generation_tokens_per_second = serving_session["average_generation_tok_per_second"]
        .as_f64()
        .unwrap_or_else(|| panic!("status must report generation throughput: {status_document}"));
    let ready_model = &status_document["configuration"]["ready_model"];
    let mtp_configured = ready_model["mtp_enabled"]["configured"]
        .as_bool()
        .unwrap_or(false);
    let mtp_effective = ready_model["mtp_enabled"]["effective"]
        .as_bool()
        .unwrap_or(false);
    let expected_mtp = phase_label == "mtp-on";
    assert_eq!(
        mtp_effective, expected_mtp,
        "phase {phase_label} must serve with mtp_enabled effective={expected_mtp}: {status_document}"
    );
    eprintln!(
        "[mtp-ab] result phase={phase_label} mtp_configured={mtp_configured} mtp_effective={mtp_effective} prefill_tok_per_second={prefill_tokens_per_second:.2} generation_tok_per_second={generation_tokens_per_second:.2}"
    );
    // The worker writes its attribution records inside the isolated home, which
    // the server owns until shutdown. Read the MTP counters while they exist.
    let attribution_log_path = isolated_development_home
        .path()
        .join(".astronomical-dev/logs/performance-attribution.jsonl");
    if let Ok(attribution_text) = std::fs::read_to_string(&attribution_log_path) {
        let mut mtp_counter_totals: std::collections::BTreeMap<String, u64> =
            std::collections::BTreeMap::new();
        let mut attribution_record_count = 0_usize;
        for line in attribution_text.lines() {
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            attribution_record_count += 1;
            if let Some(counters) = record.get("counters").and_then(Value::as_array) {
                for counter_report in counters {
                    let Some(counter_name) = counter_report.get("counter").and_then(Value::as_str)
                    else {
                        continue;
                    };
                    if counter_name.contains("mtp") {
                        let value = counter_report
                            .get("amount")
                            .and_then(Value::as_u64)
                            .unwrap_or(0);
                        *mtp_counter_totals
                            .entry(counter_name.to_owned())
                            .or_default() += value;
                    }
                }
            }
        }
        eprintln!(
            "[mtp-ab] counter phase={phase_label} attribution_records={attribution_record_count}"
        );
        for (counter_name, total) in &mtp_counter_totals {
            eprintln!("[mtp-ab] counter phase={phase_label} {counter_name}={total}");
        }
    } else {
        eprintln!(
            "[mtp-ab] counter phase={phase_label} attribution_log=unavailable path={attribution_log_path:?}"
        );
    }
    let mtp_fields = [
        "mtp_runtime_state",
        "mtp_unavailable_reason",
        "mtp_effective_execution_draft_depth",
        "mtp_capped_draft_depth",
        "mtp_resolved_requested_draft_depth",
        "mtp_depth_resolution_reason",
        "mtp_artifact_maximum_draft_depth",
        "mtp_artifact_default_draft_depth",
    ];
    for field in mtp_fields {
        eprintln!(
            "[mtp-ab] field phase={phase_label} {field}={}",
            status_document[field]
        );
    }
    // The worker log lives in the isolated home and is deleted with it. Dump
    // MTP-relevant warnings before the server stops so fallback causes survive.
    let worker_logs_directory = isolated_development_home
        .path()
        .join(".astronomical-dev/logs");
    if let Ok(worker_log_entries) = std::fs::read_dir(&worker_logs_directory) {
        for worker_log_entry in worker_log_entries.flatten() {
            let worker_log_path = worker_log_entry.path();
            if worker_log_path.extension().and_then(|e| e.to_str()) != Some("log") {
                continue;
            }
            if let Ok(worker_log_text) = std::fs::read_to_string(&worker_log_path) {
                for line in worker_log_text.lines() {
                    if line.contains("mtp")
                        || line.contains("MTP")
                        || line.contains("optional terminal history")
                    {
                        eprintln!("[mtp-ab] worker-log phase={phase_label} {line}");
                    }
                }
            }
        }
    }
    stop_serving_rest_server(rest_server).await;
}

async fn read_status_document(server_address: std::net::SocketAddr, phase_label: &str) -> Value {
    let status_response = get_endpoint(server_address, "/v1/status").await;
    let status_body = status_response
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or(status_response.as_str());
    serde_json::from_str(status_body).unwrap_or_else(|_| {
        panic!("GET /v1/status must return JSON in phase {phase_label}: {status_response}")
    })
}

/// Removes the authored MTP acceleration block so the baseline run serves the
/// same artifact with MTP off, independent of the developer's config state.
fn remove_mtp_acceleration(isolated_home_config: &std::path::Path) {
    let config_path = isolated_home_config.join(".astronomical-dev/config.json");
    let mut config_document: Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .unwrap_or_else(|error| panic!("the isolated config must be readable: {error}")),
    )
    .expect("the isolated config must be valid JSON");
    if let Some(models) = config_document
        .get_mut("models")
        .and_then(|models| models.as_object_mut())
        && let Some(model_policy) = models.get_mut(MTP_MODEL_ID)
    {
        model_policy["acceleration"]["mtp"] = Value::Null;
    }
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config_document).expect("the edited config must serialize"),
    )
    .expect("the edited isolated config must be written");
}
