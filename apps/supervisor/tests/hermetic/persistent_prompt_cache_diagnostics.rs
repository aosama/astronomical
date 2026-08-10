use std::{fs, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHealthStatus,
};
use serde_json::Value;
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::launch_test_executor_with_performance_log_directory;

#[tokio::test]
async fn should_persist_worker_cache_diagnostics_for_a_completed_user_request() {
    let performance_log_directory = tempfile::tempdir()
        .expect("the diagnostics journey should create a performance log directory");
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = launch_test_executor_with_performance_log_directory(
        worker_executable_path,
        performance_log_directory.path(),
    )
    .await
    .expect("the diagnostics fixture worker should launch");
    wait_for_ready_worker(&worker_executor).await;
    let mut stream_receiver = worker_executor
        .start_chat_generation(cache_diagnostics_command())
        .await
        .expect("the cache diagnostics request should start");

    loop {
        let stream_event = timeout(Duration::from_secs(1), stream_receiver.recv())
            .await
            .expect("the fixture should respond before the timeout")
            .expect("the fixture stream should remain open through completion");
        if matches!(stream_event, ChatGenerationStreamEvent::Completed { .. }) {
            break;
        }
    }
    assert_eq!(
        worker_executor.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );
    worker_executor
        .shutdown()
        .await
        .expect("the diagnostics fixture should stop cleanly");

    let performance_log_document =
        fs::read_to_string(performance_log_directory.path().join("performance.jsonl"))
            .expect("the completed user request should produce a performance record");
    let performance_record: Value = serde_json::from_str(
        performance_log_document
            .lines()
            .next()
            .expect("the performance log should contain one request record"),
    )
    .expect("the performance record should be valid JSON");
    let cache_diagnostics = &performance_record["persistent_prompt_cache_diagnostics"];
    assert_eq!(cache_diagnostics["lookup_outcome"], "hit");
    assert_eq!(cache_diagnostics["block_token_count"], 2_048);
    assert_eq!(cache_diagnostics["matched_sequence_state_block_count"], 3);
    assert_eq!(cache_diagnostics["restored_block_count"], 3);
    assert_eq!(
        cache_diagnostics["first_missing_sequence_state_block_index"],
        Value::Null
    );
    assert_eq!(cache_diagnostics["published_block_count"], 1);
}

async fn wait_for_ready_worker(worker_executor: &astronomical_supervisor::WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(2);
    while worker_executor.worker_health_snapshot().status != WorkerHealthStatus::Ready {
        assert!(
            Instant::now() < readiness_deadline,
            "the diagnostics fixture worker should become ready"
        );
        sleep(Duration::from_millis(20)).await;
    }
}

fn cache_diagnostics_command() -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(91_204),
        model: "astronomical/accepted-chat-fixture".to_owned(),
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
