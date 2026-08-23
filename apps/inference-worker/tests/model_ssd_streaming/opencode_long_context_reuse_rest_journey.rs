//! Repeated OpenCode-shaped REST requests proving worker availability and expert reuse.

use std::{fs, path::Path};

use serde_json::{Value, json};
use tokio::time::{Duration, Instant, interval, timeout};

use crate::model_artifact_qualification::model_artifact_rest_qualification::{
    E2E_TIMEOUT, assert_successful_streaming_chat_response,
    launch_model_artifact_rest_server_for_model_with_memory_limit, post_chat_completion,
    stop_model_artifact_rest_server,
};
use crate::model_artifact_qualification::model_artifact_rest_transport::streamed_model_text_from_chat_response;

const MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 32_000_000_000;
const MAXIMUM_QUALIFICATION_PEAK_MEMORY_BYTES: u64 = 32_320_000_000;
// This is the last historical logical-copy target, not a physical disk limit.
// Research showed that macOS may satisfy positional reads from its file cache,
// while process-attributed disk reads can also include non-expert worker input.
// Keep the value visible for trend comparison, but never let it override the
// production gates: valid output, bounded MLX memory, completion, and reuse.
const HISTORICAL_LOGICAL_EXPERT_SOURCE_READ_REFERENCE_BYTES: u64 = 25_800_000_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 5_000;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 0;
const OPENCODE_TOOL_COUNT: usize = 67;
const REQUEST_PHASE_TIMEOUT: Duration = Duration::from_secs(80);
const LONG_PROMPT_SOURCE: &str =
    include_str!("../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker for an OpenCode-shaped long-context reuse journey"]
async fn should_keep_the_worker_available_and_reuse_experts_across_repeated_opencode_long_context_requests()
 {
    timeout(E2E_TIMEOUT, run_opencode_long_context_reuse_rest_journey())
        .await
        .expect("the OpenCode worker-availability REST journey must finish within 115 seconds");
}

async fn run_opencode_long_context_reuse_rest_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the OpenCode qualification worker home should be created");
    write_opencode_qualification_config(isolated_worker_home.path(), &model_directory);
    let model_artifact_rest_server = launch_model_artifact_rest_server_for_model_with_memory_limit(
        MODEL_ID,
        model_directory,
        Some(isolated_worker_home.path()),
        None,
        Some(MAXIMUM_MLX_MEMORY_BYTES),
    )
    .await;
    let server_address = model_artifact_rest_server.server_address;

    let repeated_long_context_request_body = long_context_request_body("repeated");
    let request_phases = [
        (
            "long_context_initial",
            repeated_long_context_request_body.clone(),
        ),
        ("long_context_reuse", repeated_long_context_request_body),
    ];
    let mut failure_reason = None;
    let mut finalized_expert_payload_bytes_by_phase = Vec::new();
    for (request_phase_name, request_body) in request_phases {
        eprintln!(
            "[opencode-long-context-reuse] status=progress phase={request_phase_name} request_bytes={}",
            request_body.len()
        );
        let chat_response =
            match run_request_with_progress(server_address, request_phase_name, request_body).await
            {
                Ok(chat_response) => chat_response,
                Err(request_failure_reason) => {
                    failure_reason = Some(request_failure_reason);
                    break;
                }
            };
        if let Err(response_failure_reason) =
            check_successful_streaming_response(request_phase_name, &chat_response)
        {
            failure_reason = Some(response_failure_reason);
            break;
        }
        let ready_response =
            crate::model_artifact_qualification::model_artifact_rest_qualification::get_endpoint(
                server_address,
                "/ready",
            )
            .await;
        if !ready_response.starts_with("HTTP/1.1 200 OK") {
            failure_reason = Some(format!(
                "{request_phase_name} completed, but /ready became unhealthy: {ready_response}"
            ));
            break;
        }
        let status_document = get_status_document(server_address).await;
        let finalized_expert_payload_bytes =
            status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
                .as_u64()
                .unwrap_or(0);
        if finalized_expert_payload_bytes == 0 {
            failure_reason = Some(format!(
                "{request_phase_name} completed with no retained expert payload: {status_document}"
            ));
            break;
        }
        finalized_expert_payload_bytes_by_phase.push(finalized_expert_payload_bytes);
        eprintln!(
            "[opencode-long-context-reuse] status=progress phase={request_phase_name} worker_ready=true finalized_expert_payload_bytes={finalized_expert_payload_bytes}"
        );
    }

    stop_model_artifact_rest_server(model_artifact_rest_server).await;

    if let Some(failure_reason) = failure_reason {
        panic!("OpenCode long-context reuse journey failed: {failure_reason}");
    }
    assert_eq!(
        finalized_expert_payload_bytes_by_phase.len(),
        2,
        "both repeated long-context phases must publish retained expert payload"
    );
    assert_authoritative_paging_attribution(isolated_worker_home.path());
    eprintln!("[opencode-long-context-reuse] status=success phases=2 worker_reused=true");
}

fn write_opencode_qualification_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the OpenCode qualification configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": 32,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the OpenCode qualification configuration should serialize"),
    )
    .expect("the OpenCode qualification configuration should be written");
}

fn assert_authoritative_paging_attribution(isolated_worker_home: &Path) {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(&attribution_log_path)
        .expect("the OpenCode qualification should write performance attribution");
    let generation_reports = attribution_log
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each OpenCode qualification attribution row should be valid JSON")
        })
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .collect::<Vec<_>>();
    assert_eq!(
        generation_reports.len(),
        2,
        "both REST phases should produce generation attribution"
    );

    for (request_phase_name, generation_report) in ["long_context_initial", "long_context_reuse"]
        .into_iter()
        .zip(&generation_reports)
    {
        let phase_route_synchronization_count = summed_counter_amount(
            std::slice::from_ref(generation_report),
            "positional_file_read_call_count",
        );
        let phase_source_read_byte_count = summed_counter_amount(
            std::slice::from_ref(generation_report),
            "positional_file_read_byte_count",
        );
        let process_physical_disk_read_bytes =
            generation_report["process_physical_disk_read_bytes"]
                .as_u64()
                .expect("each generation should report process-attributed physical disk reads");
        // These numbers intentionally remain side by side. Logical source-copy
        // bytes measure Astronomical's paging work; the process counter measures
        // what macOS attributes to physical disk service over the whole request.
        // Neither metric is a substitute for request latency or worker health.
        eprintln!(
            "[opencode-long-context-reuse] status=attribution phase={request_phase_name} authoritative_route_synchronizations={phase_route_synchronization_count} logical_source_copy_bytes={phase_source_read_byte_count} process_physical_disk_read_bytes={process_physical_disk_read_bytes}"
        );
    }

    let forward_attempt_count = summed_counter_amount(
        &generation_reports,
        "gpu_resident_expert_forward_attempt_count",
    );
    let route_synchronization_count =
        summed_counter_amount(&generation_reports, "positional_file_read_call_count");
    let source_read_byte_count =
        summed_counter_amount(&generation_reports, "positional_file_read_byte_count");
    let maximum_peak_memory_bytes = generation_reports
        .iter()
        .filter_map(|generation_report| generation_report["mlx_peak_memory_bytes"].as_u64())
        .max()
        .expect("generation attribution should report peak MLX memory");
    let initial_source_read_byte_count = summed_counter_amount(
        std::slice::from_ref(&generation_reports[0]),
        "positional_file_read_byte_count",
    );
    let reused_source_read_byte_count = summed_counter_amount(
        std::slice::from_ref(&generation_reports[1]),
        "positional_file_read_byte_count",
    );

    eprintln!(
        "[opencode-long-context-reuse] status=attribution forward_attempts={forward_attempt_count} authoritative_route_synchronizations={route_synchronization_count} logical_source_copy_bytes={source_read_byte_count} historical_logical_read_reference_bytes={HISTORICAL_LOGICAL_EXPERT_SOURCE_READ_REFERENCE_BYTES} logical_read_variance_bytes={} maximum_peak_memory_bytes={maximum_peak_memory_bytes}",
        i128::from(source_read_byte_count)
            - i128::from(HISTORICAL_LOGICAL_EXPERT_SOURCE_READ_REFERENCE_BYTES)
    );
    assert_eq!(
        forward_attempt_count, 0,
        "authoritative paging must not report speculative forward attempts"
    );
    assert!(
        maximum_peak_memory_bytes <= MAXIMUM_QUALIFICATION_PEAK_MEMORY_BYTES,
        "peak MLX memory should remain within the configured ceiling plus one percent: peak_bytes={maximum_peak_memory_bytes}"
    );
    assert!(
        reused_source_read_byte_count < initial_source_read_byte_count,
        "the repeated long-context request must reuse retained expert layers and copy fewer expert bytes: initial_bytes={initial_source_read_byte_count} reused_bytes={reused_source_read_byte_count}"
    );
}

async fn get_status_document(server_address: std::net::SocketAddr) -> Value {
    let status_response =
        crate::model_artifact_qualification::model_artifact_rest_qualification::get_endpoint(
            server_address,
            "/v1/status",
        )
        .await;
    let (_, status_response_body) = status_response
        .split_once("\r\n\r\n")
        .expect("the status response should contain HTTP headers");
    serde_json::from_str(status_response_body)
        .expect("the status response body should be valid JSON")
}

fn summed_counter_amount(generation_reports: &[Value], counter_identifier: &str) -> u64 {
    generation_reports
        .iter()
        .flat_map(|generation_report| {
            generation_report["counters"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter(|counter_report| counter_report["counter"] == counter_identifier)
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

async fn run_request_with_progress(
    server_address: std::net::SocketAddr,
    request_phase_name: &str,
    request_body: String,
) -> Result<String, String> {
    let request_started_at = Instant::now();
    let mut request_task = Box::pin(tokio::spawn(post_chat_completion(
        server_address,
        request_body,
    )));
    let mut progress_interval = interval(Duration::from_secs(5));
    let request_deadline = request_started_at + REQUEST_PHASE_TIMEOUT;
    let request_timeout = tokio::time::sleep_until(request_deadline);
    tokio::pin!(request_timeout);
    progress_interval.tick().await;
    loop {
        tokio::select! {
            request_result = &mut request_task => {
                return request_result.map_err(|join_error| format!(
                    "{request_phase_name} HTTP request task failed after {:.1}s: {join_error}",
                    request_started_at.elapsed().as_secs_f64(),
                ));
            }
            _ = progress_interval.tick() => {
                let status_response = crate::model_artifact_qualification::model_artifact_rest_qualification::get_endpoint(
                    server_address,
                    "/v1/status",
                ).await;
                eprintln!(
                    "[opencode-long-context-reuse] status=progress phase={request_phase_name} elapsed_seconds={:.1} status_response={}",
                    request_started_at.elapsed().as_secs_f64(),
                    response_body_preview(&status_response),
                );
            }
            _ = &mut request_timeout => {
                request_task.as_mut().abort();
                let status_response = crate::model_artifact_qualification::model_artifact_rest_qualification::get_endpoint(
                    server_address,
                    "/v1/status",
                ).await;
                let ready_response = crate::model_artifact_qualification::model_artifact_rest_qualification::get_endpoint(
                    server_address,
                    "/ready",
                ).await;
                return Err(format!(
                    "{request_phase_name} did not return within {}s; status_response={}; ready_response={}",
                    REQUEST_PHASE_TIMEOUT.as_secs(),
                    response_body_preview(&status_response),
                    response_body_preview(&ready_response),
                ));
            }
        }
    }
}

fn response_body_preview(http_response: &str) -> String {
    http_response
        .split_once("\r\n\r\n")
        .map(|(_, response_body)| response_body)
        .unwrap_or(http_response)
        .chars()
        .take(1_000)
        .collect()
}

fn long_context_request_body(request_phase_name: &str) -> String {
    let production_shaped_tools = (0..OPENCODE_TOOL_COUNT)
        .map(|tool_number| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("opencode_qualification_tool_{tool_number}"),
                    "description": "A qualification tool that must not be called.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false,
                    },
                },
            })
        })
        .collect::<Vec<_>>();
    let source_excerpt = LONG_PROMPT_SOURCE.chars().take(10_000).collect::<String>();
    let user_prompt = format!(
        "Read the supplied Romeo and Juliet source, then reply with exactly OPENCODE_LONG_CONTEXT_OK. Request phase: {request_phase_name}. Do not call tools.\n\nSource material:\n{LONG_PROMPT_SOURCE}"
    );
    let user_prompt = user_prompt.replace(LONG_PROMPT_SOURCE, &source_excerpt);
    json!({
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a local coding assistant. Keep the answer concise."},
            {"role": "user", "content": user_prompt},
        ],
        "tools": production_shaped_tools,
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 0,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
    })
    .to_string()
}

fn check_successful_streaming_response(
    request_phase_name: &str,
    chat_response: &str,
) -> Result<(), String> {
    if chat_response.starts_with("HTTP/1.1 200 OK")
        && (chat_response.contains(r#""delta":{"content":"#)
            || chat_response.contains(r#""delta":{"reasoning_content":"#))
        && (chat_response.contains(r#""finish_reason":"length"#)
            || chat_response.contains(r#""finish_reason":"stop"#))
        && chat_response.contains("data: [DONE]")
    {
        let streamed_model_text = streamed_model_text_from_chat_response(chat_response);
        assert_successful_streaming_chat_response(chat_response);
        if streamed_model_text.is_empty() {
            return Err(format!(
                "{request_phase_name} returned a successful envelope without model text"
            ));
        }
        return Ok(());
    }

    let response_preview = chat_response.chars().take(2_000).collect::<String>();
    Err(format!(
        "{request_phase_name} returned an unsuccessful REST stream; response_preview={response_preview}"
    ))
}
