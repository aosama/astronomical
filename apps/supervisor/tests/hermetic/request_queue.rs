use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationQueueDepth, GenerationStartError,
    MlxMemoryLimitUpdateOutcome, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::launch_test_executor;

/// Second request should wait for the first to complete and then start,
/// rather than being immediately rejected with CapacityUnavailable.
///
/// This test starts Request A, spawns Request B (which should queue),
/// then completes Request A. Request B should start without any retry.
#[tokio::test]
async fn should_queue_second_request_until_first_completes() {
    let worker_executor = launch_fixture().await;

    // Request A: start a generation that takes time (delayed fixture stays active until cancelled)
    let stream_a = worker_executor
        .start_chat_generation(command_with_request_id(
            "astronomical/delayed-fragment-chat-fixture",
            1,
        ))
        .await
        .expect("first request should start immediately");

    // Request B: should queue (not reject) while Request A is active.
    // The delayed fixture stays active until cancelled, so Request B will
    // block on acquire_owned() until we cancel Request A by dropping its stream.
    let executor_for_b = worker_executor.clone();
    let queued_request_task = tokio::spawn(async move {
        executor_for_b
            .start_chat_generation(command_with_request_id("astronomical/test-worker", 2))
            .await
    });

    // Give Request B time to enter the queue (acquire queue permit and start waiting)
    sleep(Duration::from_millis(100)).await;

    // Now cancel Request A by dropping its stream. The worker will send
    // a cancellation, and the delayed fixture will acknowledge it.
    drop(stream_a);

    // Wait for the cancellation to be processed and Request B to start
    let stream_b_result = timeout(Duration::from_secs(5), queued_request_task)
        .await
        .expect("queued request should resolve within timeout")
        .expect("queued request task should not panic");

    match stream_b_result {
        Ok(mut stream_b) => {
            // Request B should now be able to produce events
            let event_b = receive_event_with_timeout(&mut stream_b, Duration::from_secs(2)).await;
            assert!(
                matches!(event_b, ChatGenerationStreamEvent::Completed { .. }),
                "queued request should complete, got {event_b:?}"
            );
        }
        Err(GenerationStartError::CapacityUnavailable) => {
            panic!("second request should queue, not be rejected with CapacityUnavailable");
        }
        Err(GenerationStartError::ModelLoadFailed {
            model_load_failure_reason,
        }) => {
            panic!(
                "the already loaded test model should not fail to load: {model_load_failure_reason}"
            );
        }
        Err(GenerationStartError::RequestTooLarge { .. }) => {
            panic!("the minimal queued request should fit the IPC frame");
        }
        Err(GenerationStartError::WorkerUnavailable) => {
            panic!("worker should be available, got WorkerUnavailable");
        }
    }

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_queue_a_memory_limit_while_generation_is_active() {
    let worker_executor = launch_fixture().await;
    let active_stream = worker_executor
        .start_chat_generation(command_with_request_id(
            "astronomical/delayed-fragment-chat-fixture",
            1,
        ))
        .await
        .expect("the delayed request should start");

    assert_eq!(
        worker_executor
            .update_mlx_memory_limit(32_000_000_000, "queued-memory-generation".to_owned())
            .await
            .expect("the active request should accept a queued limit"),
        MlxMemoryLimitUpdateOutcome::Queued
    );
    assert_eq!(
        worker_executor
            .worker_health_snapshot()
            .pending_mlx_memory_ceiling_bytes,
        Some(32_000_000_000)
    );

    drop(active_stream);
    let update_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let worker_health_snapshot = worker_executor.worker_health_snapshot();
        if worker_health_snapshot.mlx_memory_ceiling_bytes == 32_000_000_000
            && worker_health_snapshot
                .pending_mlx_memory_ceiling_bytes
                .is_none()
        {
            break;
        }
        assert!(
            Instant::now() < update_deadline,
            "the queued memory limit was not applied after finalization: {worker_health_snapshot:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
    worker_executor
        .shutdown()
        .await
        .expect("the worker should shut down");
}

/// The existing behavior of rejecting a request when capacity is full
/// should still work: when the active slot AND all queue slots are taken,
/// new requests get CapacityUnavailable immediately.
#[tokio::test]
async fn should_reject_request_when_queue_is_full() {
    let worker_executor = launch_fixture().await;

    // Start the first request (active) with the delayed fixture
    let _stream_a = worker_executor
        .start_chat_generation(command_with_request_id(
            "astronomical/delayed-fragment-chat-fixture",
            1,
        ))
        .await
        .expect("first request should start immediately");

    // Fill all queue slots
    let generation_queue_depth = GenerationQueueDepth::value();
    let mut queued_join_handles = Vec::new();
    for queue_index in 0..generation_queue_depth {
        let executor_handle = worker_executor.clone();
        let queued_request_task = tokio::spawn(async move {
            let request_id = u64::try_from(queue_index).unwrap() + 2;
            executor_handle
                .start_chat_generation(command_with_request_id(
                    "astronomical/test-worker",
                    request_id,
                ))
                .await
        });
        queued_join_handles.push(queued_request_task);
    }

    // Give queued tasks time to enter the queue (acquire queue permits)
    sleep(Duration::from_millis(200)).await;

    // The next request should be rejected immediately (queue is full)
    let rejected_result = worker_executor
        .start_chat_generation(minimal_command())
        .await;

    assert!(
        matches!(
            rejected_result,
            Err(GenerationStartError::CapacityUnavailable)
        ),
        "request beyond queue depth should be rejected with CapacityUnavailable, got {rejected_result:?}"
    );

    // Clean up: drop the active stream so the worker can process cancellations
    drop(_stream_a);

    // Give the worker time to cancel and process queued requests
    sleep(Duration::from_secs(2)).await;

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

async fn launch_fixture() -> astronomical_supervisor::WorkerHandle {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;
    worker_executor
}

fn command_with_request_id(model_id: &str, request_id_value: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id_value),
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
        qwen_thinking_channel_seed: None,
    }
}

fn minimal_command() -> ChatGenerationCommand {
    command_with_request_id("astronomical/test-worker", 1)
}

async fn receive_event_with_timeout(
    stream_receiver: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    timeout_duration: Duration,
) -> ChatGenerationStreamEvent {
    timeout(timeout_duration, stream_receiver.recv())
        .await
        .expect("the fixture should respond before the timeout")
        .expect("the stream should contain another event")
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
