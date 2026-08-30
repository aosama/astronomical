//! Isolated production supervisor and worker boundary for native FLUX acceptance.

use std::{
    fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

use astronomical_model_serving::FLUX2_KLEIN_OFFICIAL_MODEL_ID;
use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, WorkerHandle,
    build_development_application_with_reload,
};
use serde_json::Value;
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle, time::sleep};

use crate::support::http::send_http_request;

const READY_ATTEMPT_LIMIT: u8 = 70;

pub(crate) struct FluxRestServer {
    worker_handle: WorkerHandle,
    pub(crate) server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
    isolated_development_home: tempfile::TempDir,
}

pub(crate) async fn launch_flux_rest_server() -> FluxRestServer {
    let isolated_development_home = crate::support::isolated_development_home_from_user_config();
    enable_isolated_acceptance_diagnostics(&isolated_development_home);
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_development_home.path().to_path_buf(),
        production_worker_executable_path.clone(),
    );
    let resolved_runtime_config = runtime_config_resolver
        .load()
        .expect("the isolated versioned Development configuration should resolve");
    let performance_log_directory = runtime_config_resolver.instance_paths().logging_directory();
    fs::create_dir_all(&performance_log_directory)
        .expect("the isolated FLUX logging directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &production_worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the isolated FLUX performance log should open"),
        Arc::clone(&resolved_runtime_config.model_policy_catalog),
        resolved_runtime_config.worker_startup_configuration(),
    )
    .await
    .expect("the production worker should launch for FLUX acceptance");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the FLUX acceptance listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the FLUX acceptance listener should expose its address");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let application = build_development_application_with_reload(
        worker_handle.clone(),
        Arc::new(RwLock::new(resolved_runtime_config)),
        isolated_development_home.path().to_path_buf(),
    );
    let server_task = tokio::spawn(async move {
        axum::serve(listener, application)
            .with_graceful_shutdown(async {
                let _ = shutdown_receiver.await;
            })
            .await
    });
    wait_until_ready(server_address).await;
    FluxRestServer {
        worker_handle,
        server_address,
        shutdown_sender,
        server_task,
        isolated_development_home,
    }
}

impl FluxRestServer {
    pub(crate) async fn stop(self) {
        let _ = self.shutdown_sender.send(());
        self.server_task
            .await
            .expect("the FLUX REST server task should not panic")
            .expect("the FLUX REST server should stop cleanly");
        self.worker_handle
            .shutdown()
            .await
            .expect("the FLUX acceptance worker should terminate and be reaped");
    }

    pub(crate) fn attribution_log_path(&self) -> PathBuf {
        self.isolated_development_home
            .path()
            .join(".astronomical-dev/logs/performance-attribution.jsonl")
    }

    pub(crate) fn diagnostic_logs(&self) -> String {
        let logging_directory = self
            .isolated_development_home
            .path()
            .join(".astronomical-dev/logs");
        let Ok(log_entries) = fs::read_dir(logging_directory) else {
            return "acceptance log directory was unavailable".to_owned();
        };
        let mut diagnostics = String::new();
        for log_entry in log_entries.flatten() {
            let log_path = log_entry.path();
            if !log_path.is_file() {
                continue;
            }
            let Ok(log_bytes) = fs::read(&log_path) else {
                continue;
            };
            diagnostics.push_str(&format!(
                "\n--- {} ---\n",
                log_entry.file_name().to_string_lossy()
            ));
            diagnostics.push_str(&String::from_utf8_lossy(&log_bytes));
        }
        diagnostics
    }
}

pub(crate) fn assert_image_attribution(rest_server: &FluxRestServer, expected_outcomes: &[&str]) {
    let report_text = fs::read_to_string(rest_server.attribution_log_path())
        .expect("enabled isolated attribution should record FLUX reports");
    let image_reports = parse_image_attribution_reports(&report_text);
    assert_eq!(
        image_reports.len(),
        expected_outcomes.len(),
        "each image operation should emit one attribution report: {report_text}"
    );
    for (report, expected_outcome) in image_reports.iter().zip(expected_outcomes) {
        assert_eq!(report["outcome"], *expected_outcome);
        if *expected_outcome == "cancelled" {
            assert!(
                report["encoded_bytes"].is_null(),
                "cancellation must not publish partial PNG bytes"
            );
            assert!(report["operations"].as_array().is_some_and(|operations| {
                operations
                    .iter()
                    .any(|operation| operation["operation"] == "image_cancellation_synchronization")
            }));
            assert!(report["operations"].as_array().is_some_and(|operations| {
                operations
                    .iter()
                    .all(|operation| operation["operation"] != "image_final_cleanup")
            }));
        } else {
            assert!(report["operations"].as_array().is_some_and(|operations| {
                operations
                    .iter()
                    .any(|operation| operation["operation"] == "image_final_cleanup")
            }));
        }
        let final_memory = report["memory_snapshots"]
            .as_array()
            .and_then(|snapshots| {
                snapshots
                    .iter()
                    .find(|snapshot| snapshot["phase"] == "final_cleanup")
            })
            .expect("every image outcome should expose final MLX cleanup telemetry");
        assert_eq!(report["model_id"], FLUX2_KLEIN_OFFICIAL_MODEL_ID);
        let reported_revision = report["model_revision"]
            .as_str()
            .expect("image attribution should name the artifact revision");
        assert_eq!(reported_revision.len(), 40);
        assert!(
            reported_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
        assert_eq!(
            required_u64(final_memory, "mlx_allocator_cache_memory_bytes"),
            0
        );
    }
    eprintln!(
        "[flux-acceptance] phase=cleanup reports={} allocator_cache_bytes=0",
        image_reports.len()
    );
}

pub(crate) async fn wait_for_image_attribution_count(
    rest_server: &FluxRestServer,
    expected_report_count: usize,
    phase_label: &str,
) {
    for poll_count in 1..=100_u16 {
        let report_text =
            fs::read_to_string(rest_server.attribution_log_path()).unwrap_or_default();
        let report_count = parse_image_attribution_reports(&report_text).len();
        if report_count >= expected_report_count {
            eprintln!(
                "[flux-acceptance] phase={phase_label} reports={report_count} polls={poll_count}"
            );
            return;
        }
        sleep(Duration::from_millis(100)).await;
    }
    panic!("{phase_label} did not persist {expected_report_count} image attribution reports");
}

fn parse_image_attribution_reports(report_text: &str) -> Vec<Value> {
    report_text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_index, line)| {
            serde_json::from_str::<Value>(line).unwrap_or_else(|parse_error| {
                panic!(
                    "attribution JSONL line {} must contain valid JSON: {parse_error}; line={line}",
                    line_index + 1
                )
            })
        })
        .filter(|report| report["report_kind"] == "image_generation")
        .collect()
}

fn required_u64(document: &Value, field_name: &str) -> u64 {
    document[field_name]
        .as_u64()
        .unwrap_or_else(|| panic!("{field_name} must contain numeric memory telemetry: {document}"))
}

pub(crate) async fn get_status(server_address: SocketAddr) -> serde_json::Value {
    let response = send_http_request(
        server_address,
        format!("GET /v1/status HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n"),
    )
    .await;
    response_json(&response)
}

pub(crate) async fn post_image(server_address: SocketAddr, request_body: String) -> String {
    send_http_request(
        server_address,
        format!(
            "POST /v1/images/generations HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
            request_body.len()
        ),
    )
    .await
}

pub(crate) fn response_json(response: &str) -> serde_json::Value {
    let (_, response_body) = response
        .split_once("\r\n\r\n")
        .expect("the HTTP response should contain headers");
    serde_json::from_str(response_body)
        .unwrap_or_else(|_| panic!("the HTTP response should contain JSON: {response}"))
}

fn enable_isolated_acceptance_diagnostics(isolated_home: &tempfile::TempDir) {
    let config_path = isolated_home.path().join(".astronomical-dev/config.json");
    let mut config_document: serde_json::Value = serde_json::from_slice(
        &fs::read(&config_path).expect("the copied Development config should be readable"),
    )
    .expect("the copied Development config should contain JSON");
    config_document["diagnostics"]["performance_attribution_enabled"] = serde_json::json!(true);
    config_document["prompt_cache"]["enabled"] = serde_json::json!(false);
    fs::write(
        config_path,
        serde_json::to_vec_pretty(&config_document)
            .expect("the isolated Development config should serialize"),
    )
    .expect("the isolated Development config should remain writable");
}

async fn wait_until_ready(server_address: SocketAddr) {
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let status = get_status(server_address).await;
        if status["status"] == "ready" {
            eprintln!("[flux-acceptance] phase=worker-ready attempt={readiness_attempt}");
            return;
        }
        eprintln!(
            "[flux-acceptance] phase=worker-starting attempt={readiness_attempt}/{READY_ATTEMPT_LIMIT}"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the FLUX acceptance worker did not become ready before the deadline");
}
