use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationFailureReason,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, ExpertMemoryMode, MAX_IPC_FRAME_BYTES,
    MlxMemorySnapshotSource, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    GenerationStartError, WorkerActivity, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

use crate::common::supervisor::launch_test_executor;

const RETIRED_SMALL_FRAME_BYTES: usize = 64 * 1024;

#[tokio::test]
async fn should_stream_ordered_chat_outputs() {
    let worker_executor = launch_fixture().await;
    let mut stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/accepted-chat-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the chat request");

    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::ReasoningFragment("accepted chat reasoning".to_owned())
    );
    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::TextFragment("accepted chat text".to_owned())
    );
    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "read".to_owned(),
            arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
        }
    );
    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 1,
            function_name: "glob".to_owned(),
            arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
        }
    );
    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 2,
            generated_token_count: 4,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::ToolCalls,
        }
    );
    assert_eq!(
        worker_executor.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_replace_prefill_memory_with_finalized_residency_memory() {
    let worker_executor = launch_fixture().await;
    let mut stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/prefill-progress-fixture",
            1,
        ))
        .await
        .expect("the fixture should start the prefill progress request");

    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::PrefillProgress {
            processed_tokens: 2_048,
            total_tokens: 50_000,
            elapsed_millis: 1_500,
            forward_prefill_chunk_elapsed_millis: Some(1_400),
            completed_prefill_chunk_tokens: Some(2_048),
            mlx_active_memory_bytes: Some(11_000),
            mlx_allocator_cache_memory_bytes: Some(12_000),
            mlx_peak_memory_bytes: Some(13_000),
        }
    );
    assert!(matches!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::TextFragment(_)
    ));
    let completion_event = receive_event(&mut stream_receiver).await;
    assert!(
        matches!(
            completion_event,
            ChatGenerationStreamEvent::Completed { .. }
        ),
        "expected completion after prefill progress and output, got {completion_event:?}"
    );
    let worker_health_snapshot = worker_executor.worker_health_snapshot();
    let latest_mlx_memory_snapshot = worker_health_snapshot
        .latest_mlx_memory_snapshot
        .expect("finalized telemetry should replace prefill telemetry");
    assert_eq!(
        latest_mlx_memory_snapshot.source,
        MlxMemorySnapshotSource::Finalized
    );
    assert_eq!(latest_mlx_memory_snapshot.expert_payload_bytes, 19_000);
    assert_eq!(latest_mlx_memory_snapshot.model_core_payload_bytes, 3_000);
    assert_eq!(latest_mlx_memory_snapshot.context_state_payload_bytes, 0);
    assert_eq!(latest_mlx_memory_snapshot.active_memory_bytes, 24_000);
    assert_eq!(
        worker_health_snapshot.expert_memory_mode,
        Some(ExpertMemoryMode::Resident)
    );
    assert_eq!(
        worker_health_snapshot
            .recent_prompt_processing_chunk_optimization_outcomes
            .len(),
        1
    );
    assert_eq!(
        worker_health_snapshot.recent_prompt_processing_chunk_optimization_outcomes[0]
            .selected_candidate_chunk_size_tokens,
        4_096
    );
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_track_generation_activity_without_worker_phase_events() {
    let worker_executor = launch_fixture().await;
    let mut stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/activity-transition-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the activity transition request");

    wait_for_activity(&worker_executor, WorkerActivity::PromptProcessing).await;
    assert!(matches!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::TextFragment(_)
    ));
    wait_for_activity(&worker_executor, WorkerActivity::Generating).await;
    assert!(matches!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));
    wait_for_activity(&worker_executor, WorkerActivity::Idle).await;

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_report_malformed_output_and_reuse_the_worker() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/malformed-output-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the malformed-output request");
    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Failed {
            reason: ChatGenerationFailureReason::MalformedModelOutput,
        }
    );

    let mut followup_stream = worker_executor
        .start_chat_generation(minimal_command())
        .await
        .expect("the same worker should start a follow-up request");
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
async fn should_send_large_but_bounded_chat_commands_to_the_worker_in_one_ipc_frame() {
    let worker_executor = launch_fixture().await;
    let mut large_command = minimal_command();
    large_command.request_id = RequestId::new(810);
    large_command.messages = vec![ChatMessage::User {
        content: "x".repeat(RETIRED_SMALL_FRAME_BYTES * 2),
        images: Vec::new(),
    }];
    let serialized_command_bytes = serde_json::to_vec(&large_command)
        .expect("the large chat command should serialize")
        .len();
    assert!(serialized_command_bytes > RETIRED_SMALL_FRAME_BYTES);
    assert!(serialized_command_bytes <= MAX_IPC_FRAME_BYTES);
    let mut stream_receiver = worker_executor
        .start_chat_generation(large_command)
        .await
        .expect("the fixture should start the large chat request");

    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        }
    );
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_terminate_worker_after_an_out_of_order_event() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/out-of-order-chat-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the malformed-order request");
    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_terminate_worker_after_duplicate_generation_preparation() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/duplicate-generation-preparation-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the duplicate-preparation request");

    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_terminate_worker_after_an_empty_output_batch() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/empty-output-batch-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the empty-batch request");

    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_validate_an_entire_output_batch_before_forwarding_any_entry() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/invalid-output-batch-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the invalid-batch request");

    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_terminate_worker_after_an_over_budget_tool_completion() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/over-budget-tool-completion-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the over-budget completion request");
    assert!(matches!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::ToolCall { .. }
    ));
    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_terminate_worker_after_an_unsolicited_cancellation() {
    let worker_executor = launch_fixture().await;
    let mut malformed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/unsolicited-cancellation-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the unsolicited-cancellation request");
    assert_eq!(
        receive_event(&mut malformed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
    wait_for_health(&worker_executor, WorkerHealthStatus::Unavailable).await;
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_shut_down_when_the_http_stream_stops_consuming_output() {
    let worker_executor = launch_fixture().await;
    let mut backpressure_command = command_for_model("astronomical/backpressure-fixture", 5_000);
    backpressure_command.settings.max_output_tokens = 1;
    let stream_receiver = worker_executor
        .start_chat_generation(backpressure_command)
        .await
        .expect("the fixture should start the backpressure request");
    sleep(Duration::from_millis(200)).await;

    let shutdown_outcome = timeout(Duration::from_secs(1), worker_executor.shutdown()).await;
    drop(stream_receiver);
    shutdown_outcome
        .expect("shutdown must not block behind an undrained HTTP stream")
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_cancel_a_disconnected_stream_and_reuse_capacity() {
    let worker_executor = launch_fixture().await;
    let stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/delayed-fragment-chat-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the delayed request");
    drop(stream_receiver);

    let followup_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        match worker_executor
            .start_chat_generation(minimal_command())
            .await
        {
            Ok(mut followup_stream) => {
                assert!(matches!(
                    receive_event(&mut followup_stream).await,
                    ChatGenerationStreamEvent::Completed { .. }
                ));
                break;
            }
            Err(GenerationStartError::CapacityUnavailable)
                if Instant::now() < followup_deadline =>
            {
                sleep(Duration::from_millis(20)).await;
            }
            Err(start_error) => panic!("follow-up request failed: {start_error:?}"),
        }
    }
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_fail_one_stream_when_the_worker_exits() {
    let worker_executor = launch_fixture().await;
    let mut stream_receiver = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/exit-after-chat-admission-fixture",
            1_000,
        ))
        .await
        .expect("the fixture should start the exit request");
    assert_eq!(
        receive_event(&mut stream_receiver).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );
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

fn minimal_command() -> ChatGenerationCommand {
    command_for_model("astronomical/test-worker", 1_000)
}

fn command_for_model(model_id: &str, _timeout_millis: u64) -> ChatGenerationCommand {
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

async fn wait_for_activity(
    worker_executor: &astronomical_supervisor::WorkerHandle,
    expected_worker_activity: WorkerActivity,
) {
    let activity_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let observed_worker_activity = worker_executor.worker_health_snapshot().activity;
        if observed_worker_activity == expected_worker_activity {
            return;
        }
        assert!(
            Instant::now() < activity_deadline,
            "worker activity remained {observed_worker_activity:?}"
        );
        sleep(Duration::from_millis(20)).await;
    }
}
