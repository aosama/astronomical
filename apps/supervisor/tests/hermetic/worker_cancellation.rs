use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerControlError, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::{
    launch_test_executor, launch_test_executor_with_cancellation_acknowledgement_timeout,
};

#[tokio::test]
async fn should_replace_a_worker_that_does_not_acknowledge_cancellation() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor_with_cancellation_acknowledgement_timeout(
        worker_executable_path,
        Duration::from_millis(100),
    )
    .await
    .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/unacknowledged-cancellation-fixture",
        ))
        .await
        .expect("the fixture should start the cancellable request");

    drop(stream_receiver);
    sleep(Duration::from_millis(300)).await;
    timeout(
        Duration::from_secs(5),
        wait_for_health(&worker_executor, WorkerHealthStatus::Ready),
    )
    .await
    .expect("the explicit cancellation timeout should bound worker replacement");
    assert_worker_accepts_followup_request(&worker_executor).await;

    worker_executor
        .shutdown()
        .await
        .expect("shutdown after cancellation containment should be idempotent");
}

#[tokio::test]
async fn should_keep_worker_ready_when_prefill_cancellation_acknowledgement_is_delayed() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/delayed-cancellation-acknowledgement-fixture",
        ))
        .await
        .expect("the fixture should start the delayed-cancellation request");

    drop(stream_receiver);
    sleep(Duration::from_secs(5)).await;

    assert_eq!(
        worker_executor.worker_health_snapshot().status,
        WorkerHealthStatus::Ready,
        "a bounded slow prefill cancellation must not permanently disable the worker"
    );
    let mut followup_stream = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await
        .expect("the same worker should accept a request after delayed cancellation");
    assert!(matches!(
        receive_event(&mut followup_stream).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_replace_a_worker_after_an_unexpected_cancellation_event() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/unexpected-cancellation-event-fixture",
        ))
        .await
        .expect("the fixture should start the cancellable request");

    drop(stream_receiver);
    sleep(Duration::from_millis(200)).await;
    timeout(
        Duration::from_secs(5),
        wait_for_health(&worker_executor, WorkerHealthStatus::Ready),
    )
    .await
    .expect("the unexpected cancellation event should replace the worker");
    assert_worker_accepts_followup_request(&worker_executor).await;

    let diagnostic_error = WorkerControlError::UnexpectedCancellationEvent {
        request_id: 1,
        unexpected_worker_event_summary: "completed request_id=2".to_owned(),
    };
    assert_eq!(
        diagnostic_error.to_string(),
        "worker emitted an unexpected event while cancelling request 1: completed request_id=2"
    );

    worker_executor
        .shutdown()
        .await
        .expect("shutdown after cancellation containment should be idempotent");
}

#[tokio::test]
async fn should_keep_worker_ready_when_cancellation_publishes_prompt_cache_stats() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/cache-stats-during-cancellation-fixture",
        ))
        .await
        .expect("the fixture should start the cancellable request");

    drop(stream_receiver);
    sleep(Duration::from_millis(200)).await;

    let health_snapshot = worker_executor.worker_health_snapshot();
    assert_eq!(
        health_snapshot.status,
        WorkerHealthStatus::Ready,
        "prompt-cache stats during cancellation must not disable the worker"
    );
    let published_cache_stats =
        astronomical_supervisor::PersistentPromptCacheSummary::from_worker_event(
            health_snapshot.persistent_prompt_cache_stats.as_ref(),
        );
    assert_eq!(published_cache_stats.hits, 1);
    assert_eq!(published_cache_stats.tokens_saved, 2_048);

    let mut followup_stream = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await
        .expect("the same worker should accept a request after cache-stats cancellation");
    assert!(matches!(
        receive_event(&mut followup_stream).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_keep_worker_ready_when_cancellation_publishes_mlx_memory() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/mlx-memory-during-cancellation-fixture",
        ))
        .await
        .expect("the fixture should start the cancellable request");

    drop(stream_receiver);
    wait_for_active_memory(&worker_executor, Some(44_000)).await;

    assert_eq!(
        worker_executor.worker_health_snapshot().status,
        WorkerHealthStatus::Ready,
        "MLX memory telemetry during cancellation must not disable the worker"
    );
    assert_worker_accepts_followup_request(&worker_executor).await;

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_keep_worker_ready_when_cancellation_clears_mlx_memory() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/mlx-memory-clear-during-cancellation-fixture",
        ))
        .await
        .expect("the fixture should start the cancellable request");
    wait_for_active_memory(&worker_executor, Some(33_000)).await;

    drop(stream_receiver);
    wait_for_active_memory(&worker_executor, None).await;

    assert_eq!(
        worker_executor.worker_health_snapshot().status,
        WorkerHealthStatus::Ready,
        "cleared MLX memory telemetry during cancellation must not disable the worker"
    );
    assert_worker_accepts_followup_request(&worker_executor).await;

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

fn command_for_model(model_id: &str) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: "hello".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}

async fn receive_event(
    stream_receiver: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
) -> ChatGenerationStreamEvent {
    timeout(Duration::from_secs(1), stream_receiver.recv())
        .await
        .expect("the fixture should respond before the timeout")
        .expect("the stream should contain another event")
}

async fn assert_worker_accepts_followup_request(
    worker_executor: &astronomical_supervisor::WorkerHandle,
) {
    let mut followup_stream = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await
        .expect("the same worker should accept a request after cancellation telemetry");
    assert!(matches!(
        receive_event(&mut followup_stream).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));
}

async fn wait_for_active_memory(
    worker_executor: &astronomical_supervisor::WorkerHandle,
    expected_active_memory_bytes: Option<u64>,
) {
    let memory_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let observed_active_memory_bytes = worker_executor
            .worker_health_snapshot()
            .latest_mlx_memory_snapshot
            .map(|memory_snapshot| memory_snapshot.active_memory_bytes);
        if observed_active_memory_bytes == expected_active_memory_bytes {
            return;
        }
        assert!(
            Instant::now() < memory_deadline,
            "worker memory remained {observed_active_memory_bytes:?}"
        );
        sleep(Duration::from_millis(20)).await;
    }
}

async fn wait_for_health(
    worker_executor: &astronomical_supervisor::WorkerHandle,
    expected_status: WorkerHealthStatus,
) {
    let health_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let observed_status = worker_executor.worker_health_snapshot().status;
        if observed_status == expected_status {
            return;
        }
        assert!(
            Instant::now() < health_deadline,
            "worker health remained {observed_status:?}"
        );
        sleep(Duration::from_millis(20)).await;
    }
}
