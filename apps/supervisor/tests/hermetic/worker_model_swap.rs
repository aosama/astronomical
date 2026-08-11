//! Acceptance coverage for queued generation across a lazy model swap.
//!
//! The success journey proves idle telemetry can be observed before the swap
//! acknowledgement without losing the queued request. The failure journey proves
//! that sharing the event handler does not weaken generation correlation rules.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    GenerationStartError, WorkerHandle, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 20_480;
const DELAYED_COMPLETION_MODEL_ID: &str = "astronomical/delayed-completion-model";
const GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: &str =
    "astronomical/generation-event-before-swap-model";
const TELEMETRY_BEFORE_SWAP_MODEL_ID: &str = "astronomical/telemetry-before-swap-model";

#[tokio::test]
async fn should_complete_a_queued_model_swap_when_idle_telemetry_arrives_before_acknowledgement() {
    let worker_handle = launch_idle_worker_fixture().await;
    let mut first_generation_events = worker_handle
        .start_chat_generation(chat_command(DELAYED_COMPLETION_MODEL_ID, 1))
        .await
        .expect("the first model should load and start generation");

    let queued_worker_handle = worker_handle.clone();
    let queued_generation_task = tokio::spawn(async move {
        queued_worker_handle
            .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 2))
            .await
    });

    assert_generation_completed(&mut first_generation_events).await;
    let mut queued_generation_events = timeout(Duration::from_secs(2), queued_generation_task)
        .await
        .expect("the queued model swap should finish before the timeout")
        .expect("the queued generation task should not panic")
        .expect("idle telemetry must not make the queued model swap unavailable");
    assert_generation_completed(&mut queued_generation_events).await;

    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(worker_health_snapshot.status, WorkerHealthStatus::Ready);
    assert_eq!(
        worker_health_snapshot.ready_model_id.as_deref(),
        Some(TELEMETRY_BEFORE_SWAP_MODEL_ID)
    );
    worker_handle
        .shutdown()
        .await
        .expect("the worker should remain available for graceful shutdown");
}

#[tokio::test]
async fn should_reject_a_generation_scoped_event_while_waiting_for_model_swap() {
    let worker_handle = launch_idle_worker_fixture().await;

    let generation_start_outcome = worker_handle
        .start_chat_generation(chat_command(GENERATION_EVENT_BEFORE_SWAP_MODEL_ID, 3))
        .await;

    assert!(matches!(
        generation_start_outcome,
        Err(GenerationStartError::WorkerUnavailable)
    ));
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Unavailable
    );
    worker_handle
        .shutdown()
        .await
        .expect("the contained worker should already be reaped");
}

async fn launch_idle_worker_fixture() -> WorkerHandle {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_log_directory =
        tempfile::tempdir().expect("test performance log directory should be created");
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        GenerationPerformanceLog::open(temporary_log_directory.path())
            .expect("test performance log should be created"),
        Arc::new(HashMap::from([
            (
                DELAYED_COMPLETION_MODEL_ID.to_owned(),
                PathBuf::from("/models/delayed-completion-model"),
            ),
            (
                TELEMETRY_BEFORE_SWAP_MODEL_ID.to_owned(),
                PathBuf::from("/models/telemetry-before-swap-model"),
            ),
            (
                GENERATION_EVENT_BEFORE_SWAP_MODEL_ID.to_owned(),
                PathBuf::from("/models/generation-event-before-swap-model"),
            ),
        ])),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_ready_worker(&worker_handle).await;
    worker_handle
}

async fn wait_for_ready_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let worker_health_status = worker_handle.worker_health_snapshot().status;
        if worker_health_status == WorkerHealthStatus::Ready {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "idle worker did not become ready; last status was {worker_health_status:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn assert_generation_completed(
    generation_events: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
) {
    let generation_event = timeout(Duration::from_secs(2), generation_events.recv())
        .await
        .expect("the generation should complete before the timeout")
        .expect("the generation stream should contain a completion event");
    assert!(matches!(
        generation_event,
        ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        }
    ));
}

fn chat_command(model_id: &str, request_id: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: "Wherefore art thou Romeo?".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}
