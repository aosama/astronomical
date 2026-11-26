//! Server side of the daemon IPC journeys: stub executors, services started
//! on a fresh instance state directory, and their shutdown and cleanup.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use astronomical_daemon::{
    DaemonIpcService, DaemonPerformanceAttributionLog, WorkerHealthSnapshot,
};
use config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance, DiscoveredModel};
use ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DaemonRequest, DaemonResponse, DaemonTransportError,
    DaemonWorkerStatus,
};
use tokio::time::timeout;

use crate::support::daemon_ipc::{
    HANDSHAKE_TEST_TIMEOUT, StubGenerationExecutor, fresh_instance_state_directory,
    ipc_runtime_config, ready_stub_executor, start_stub_daemon_ipc_service,
    start_stub_daemon_ipc_service_with_runtime_config, unavailable_generation_context,
};

/// A stub whose worker reports the given health and streams nothing, for
/// journeys that stop at the daemon gate or only read worker health.
pub(crate) fn stub_executor_with_health_snapshot(
    health_snapshot: WorkerHealthSnapshot,
) -> Arc<StubGenerationExecutor> {
    Arc::new(StubGenerationExecutor::with_health_snapshot(
        health_snapshot,
    ))
}

/// Starts a daemon whose reloadable config knows the given installed models
/// and whose policy catalog holds the given requestable model ids.
pub(crate) async fn start_service_with_installed_models(
    state_directory: &Path,
    stub_executor: Arc<StubGenerationExecutor>,
    installed_models: Vec<DiscoveredModel>,
    policy_catalog_model_ids: &[&str],
) -> DaemonIpcService {
    let reloadable_config = Arc::new(RwLock::new(ipc_runtime_config(
        installed_models,
        policy_catalog_model_ids,
    )));
    start_stub_daemon_ipc_service_with_runtime_config(
        state_directory,
        stub_executor,
        Some(reloadable_config),
    )
    .await
}

pub(crate) async fn shut_down_service(daemon_ipc_service: DaemonIpcService) {
    timeout(HANDSHAKE_TEST_TIMEOUT, daemon_ipc_service.shutdown())
        .await
        .expect("the daemon IPC service shutdown should finish inside the test timeout")
        .expect("the daemon IPC service should shut down cleanly");
}

pub(crate) async fn shut_down_and_remove_state_directory(
    daemon_ipc_service: DaemonIpcService,
    state_directory: &Path,
) {
    shut_down_service(daemon_ipc_service).await;
    let _ = std::fs::remove_dir_all(state_directory);
}

pub(crate) fn development_instance_paths(state_directory: &Path) -> AstronomicalInstancePaths {
    AstronomicalInstancePaths::for_state_directory(
        state_directory.to_path_buf(),
        AstronomicalRuntimeInstance::Development,
    )
}

/// Starts the service the way the daemon binary does, on the instance socket
/// under the given paths, with a worker that is not available. Returns the
/// start outcome so a journey can also assert that a second start is refused.
pub(crate) async fn try_start_service_on_instance_socket(
    instance_paths: &AstronomicalInstancePaths,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
) -> Result<DaemonIpcService, DaemonTransportError> {
    timeout(
        HANDSHAKE_TEST_TIMEOUT,
        astronomical_daemon::start_daemon_ipc_service(
            instance_paths,
            unavailable_generation_context(),
            daemon_attribution_log,
        ),
    )
    .await
    .expect("the daemon IPC service start should finish inside the test timeout")
}

/// A daemon whose worker is ready with the chat model every IPC journey
/// uses, started in a fresh state directory the caller removes at the end.
pub(crate) async fn start_ready_chat_service(
    test_name: &str,
) -> (DaemonIpcService, Arc<StubGenerationExecutor>, PathBuf) {
    let state_directory = fresh_instance_state_directory(test_name);
    let stub_executor = ready_stub_executor("test/local-chatter");
    let daemon_ipc_service =
        start_stub_daemon_ipc_service(&state_directory, Arc::clone(&stub_executor)).await;
    (daemon_ipc_service, stub_executor, state_directory)
}

/// The status answer of a ready worker that has served nothing and is not
/// loading, for the given resident and default model ids.
pub(crate) fn ready_idle_status_response(
    ready_model_id: &str,
    default_model_id: &str,
) -> DaemonResponse {
    DaemonResponse::Status {
        worker_status: DaemonWorkerStatus::Ready,
        ready_model_id: Some(ready_model_id.to_owned()),
        default_model_id: Some(default_model_id.to_owned()),
        model_load_in_flight: None,
        request_progress_count: 0,
        neural_engine_prefill: None,
        draft_model: None,
        last_reply_drafting: None,
    }
}

/// A "Say hello" chat request with every sampling setting left unset, so a
/// journey varies only the output budget and the schema it is about.
pub(crate) fn say_hello_chat_request(
    model: &str,
    max_output_tokens: u16,
    schema_json: Option<String>,
) -> DaemonRequest {
    DaemonRequest::ChatGenerate {
        model: model.to_owned(),
        messages: vec![ChatMessage::User {
            content: "Say hello".to_owned(),
            images: vec![],
        }],
        settings: ChatGenerationSettings {
            max_output_tokens,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        schema_json,
    }
}
