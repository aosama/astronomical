//! Whole-product check of the memory split shown by the public status surface.

use std::{fs, net::SocketAddr};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use serde_json::{Value, json};
use serial_test::serial;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::{Duration, Instant, sleep, timeout},
};

use crate::common::daemon;

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real_models=true and the large sparse artifact; launches the real daemon and worker at 37 GB"]
#[serial]
async fn should_report_complete_decode_residency_through_the_real_daemon_at_37_gigabytes() {
    timeout(JOURNEY_BUDGET, run_full_fit_status_journey())
        .await
        .expect("the whole-product full-fit journey must stay within the 120-second runner budget");
}

#[expect(clippy::too_many_lines)]
async fn run_full_fit_status_journey() {
    let context = TestRunContext::current();
    assert!(
        context.configuration.real_models,
        "this journey must not substitute a scripted worker"
    );
    let model_id = astronomical_test_harness::fixtures::large_sparse_moe_streaming_model_id();
    let model_directory = context.configuration.model_directory.unwrap_or_else(|| {
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("the Development configuration should load")
            .find_configured_model_directory_by_id(model_id)
            .expect("the model directories should be searchable")
            .expect("the large sparse artifact should be installed")
    });
    let evidence_directory = context
        .configuration
        .evidence_directory
        .expect("run the real-model journey through cargo xtask test-run to retain its evidence");
    fs::create_dir_all(&evidence_directory).expect("the evidence directory should exist");
    // Deep runner evidence paths exceed the Unix-domain socket path limit.
    // Keep live state in the platform temporary directory and copy logs below.
    let isolated_state = tempfile::tempdir().expect("the isolated daemon state should exist");
    let state_directory = isolated_state.path();
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&sparse_decode_full_fit::configuration_document(
            &model_directory,
        ))
        .expect("the configuration should serialize"),
    )
    .expect("the configuration should persist");
    let daemon_path = astronomical_test_harness::workspace_binary("astronomicald");
    let (mut daemon_process, address) = daemon::spawn_actual_instance_daemon(
        daemon_path
            .to_str()
            .expect("the executable path should be UTF-8"),
        "development",
        state_directory,
    )
    .await;
    let mut daemon_stderr = daemon_process
        .stderr
        .take()
        .expect("daemon diagnostics should be piped");
    let diagnostics_task = tokio::spawn(async move {
        let mut diagnostics = String::new();
        daemon_stderr
            .read_to_string(&mut diagnostics)
            .await
            .expect("daemon diagnostics should be readable");
        diagnostics
    });
    daemon::wait_until_endpoint_contains(
        address,
        "/v1/status",
        "\"status\":\"ready\"",
        Duration::from_secs(20),
    )
    .await;
    let advertised_models = daemon::get_endpoint(address, "/v1/models").await;
    assert!(
        advertised_models.contains(model_id),
        "the real daemon must discover the requested artifact: {advertised_models}"
    );
    let request_outcome = timeout(Duration::from_secs(105), async {
        tokio::try_join!(
            complete_romeo_and_juliet(address, model_id),
            observe_decode_status(address)
        )
    })
    .await;
    daemon::terminate_daemon(&daemon_process);
    assert!(
        daemon_process
            .wait()
            .await
            .expect("the daemon should be reaped")
            .success()
    );
    fs::write(
        evidence_directory.join("daemon-diagnostics.log"),
        diagnostics_task.await.expect("diagnostics should complete"),
    )
    .expect("daemon diagnostics should persist");
    let retained_logging_directory = evidence_directory.join("daemon-logs");
    fs::create_dir(&retained_logging_directory).expect("the retained log directory should exist");
    for logging_entry in fs::read_dir(state_directory.join("logs"))
        .expect("the daemon logging directory should be readable")
    {
        let logging_entry = logging_entry.expect("each daemon log should be accessible");
        fs::copy(
            logging_entry.path(),
            retained_logging_directory.join(logging_entry.file_name()),
        )
        .expect("daemon and worker logs should persist");
    }
    let (completion, statuses) = request_outcome
        .expect("the request must finish within its budget; daemon logs were retained")
        .unwrap_or_else(|failure| panic!("{failure}; daemon logs were retained"));
    fs::write(
        evidence_directory.join("decode-status-samples.json"),
        serde_json::to_vec_pretty(&statuses).expect("status evidence should serialize"),
    )
    .expect("status evidence should persist");
    assert!(
        completion.contains("\"finish_reason\""),
        "the completion stream must finish"
    );
    assert!(
        !statuses.is_empty(),
        "the whole-product journey must observe generating status"
    );
    for status in &statuses {
        sparse_decode_full_fit::validate_decode_residency(status)
            .unwrap_or_else(|failure| panic!("{failure}"));
    }
    let report =
        sparse_decode_full_fit::read_generation_attribution_report(&retained_logging_directory, 1)
            .expect("the real worker must persist generation attribution");
    sparse_decode_full_fit::validate_generation_attribution(&report)
        .unwrap_or_else(|failure| panic!("{failure}"));
}

async fn complete_romeo_and_juliet(address: SocketAddr, model_id: &str) -> Result<String, String> {
    let source = include_str!(
        "../../../acceptance/tests/fixtures/model_metrics_50000_romeo_and_juliet_words.txt"
    );
    let source = source
        .split_whitespace()
        .take(18_000)
        .collect::<Vec<_>>()
        .join(" ");
    let request_body = serde_json::to_string(&json!({
        "model": model_id,
        "messages": [{"role": "user", "content": format!("Continue Romeo and Juliet.\n\n{source}")}],
        "stream": true,
        "max_tokens": 128,
        "thinking_budget": 32,
        "temperature": 1
    })).expect("the request should serialize");
    let mut connection = TcpStream::connect(address)
        .await
        .expect("the daemon should accept a request");
    connection.write_all(format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    ).as_bytes()).await.expect("the request should send");
    let mut completion = String::new();
    connection
        .read_to_string(&mut completion)
        .await
        .expect("the stream should remain readable");
    if !completion.starts_with("HTTP/1.1 200") {
        return Err(format!("the real daemon rejected generation: {completion}"));
    }
    Ok(completion)
}

async fn observe_decode_status(address: SocketAddr) -> Result<Vec<Value>, String> {
    let started_at = Instant::now();
    let mut last_progress_at = started_at;
    let mut saw_request = false;
    let mut statuses = Vec::new();
    loop {
        let endpoint = daemon::get_endpoint(address, "/v1/status").await;
        let (_, body) = endpoint
            .split_once("\r\n\r\n")
            .expect("status should contain HTTP headers");
        let status: Value = serde_json::from_str(body).expect("the status body should parse");
        saw_request |= status["activity"] != "idle";
        if status["activity"] == "generating"
            && status["mlx_memory_snapshot"]["source"] == "decode_submitted"
        {
            statuses.push(status.clone());
        }
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            let elapsed_seconds = started_at.elapsed().as_secs();
            eprintln!(
                "[decode-full-fit-product] progress_percent={} eta_seconds={} activity={}",
                elapsed_seconds * 100 / JOURNEY_BUDGET.as_secs(),
                JOURNEY_BUDGET.as_secs().saturating_sub(elapsed_seconds),
                status["activity"]
            );
            last_progress_at = Instant::now();
        }
        if saw_request && status["activity"] == "idle" {
            return Ok(statuses);
        }
        assert!(
            started_at.elapsed() < JOURNEY_BUDGET,
            "generation did not return idle: {status}"
        );
        sleep(Duration::from_millis(50)).await;
    }
}
