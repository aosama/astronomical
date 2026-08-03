use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::{
    launch_test_executor, launch_test_executor_with_cancellation_acknowledgement_timeout,
};

#[tokio::test]
async fn should_contain_a_worker_that_does_not_acknowledge_cancellation() {
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
    timeout(
        Duration::from_secs(5),
        wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable),
    )
    .await
    .expect("the explicit cancellation timeout should bound cancellation cleanup");

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
