//! A prompt-token count through the real worker loop and IPC framing: the
//! daemon must dispatch `CountChatTokens`, not `Generate`, and must release
//! the queue afterwards. The scripted worker answers a count with seven
//! tokens per message, a figure no `Generate` reply carries.

use std::time::Duration;

use astronomical_daemon::{ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHealthStatus};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, RequestId,
};
use tokio::time::{Instant, sleep, timeout};

use crate::support::daemon;

const SCRIPTED_TOKENS_PER_MESSAGE: u32 = 7;

#[tokio::test]
async fn should_dispatch_a_count_and_return_the_worker_measured_prompt_length() {
    let worker_executor = launch_ready_fixture().await;

    let counted_event = count_tokens(&worker_executor, three_message_command(41)).await;

    assert_eq!(
        counted_event,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 3 * SCRIPTED_TOKENS_PER_MESSAGE,
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
async fn should_release_the_queue_after_a_count_so_the_next_request_runs() {
    let worker_executor = launch_ready_fixture().await;
    let _first_count = count_tokens(&worker_executor, three_message_command(42)).await;

    let second_count = count_tokens(&worker_executor, three_message_command(43)).await;

    assert!(matches!(
        second_count,
        ChatGenerationStreamEvent::Completed {
            prompt_token_count: 21,
            ..
        }
    ));
    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

async fn count_tokens(
    worker_executor: &astronomical_daemon::WorkerHandle,
    count_command: ChatGenerationCommand,
) -> ChatGenerationStreamEvent {
    let (admission_sender, _admission_receiver) = tokio::sync::oneshot::channel();
    let mut stream_receiver = worker_executor
        .start_chat_token_count_with_admission_signal(count_command, admission_sender)
        .await
        .expect("the fixture should accept the count");
    timeout(Duration::from_secs(1), stream_receiver.recv())
        .await
        .expect("the fixture should count before the timeout")
        .expect("the count stream should carry its answer")
}

async fn launch_ready_fixture() -> astronomical_daemon::WorkerHandle {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-daemon-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = daemon::launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    let health_deadline = Instant::now() + Duration::from_secs(5);
    while worker_executor.worker_health_snapshot().status != WorkerHealthStatus::Ready {
        assert!(
            Instant::now() < health_deadline,
            "the worker never became ready"
        );
        sleep(Duration::from_millis(20)).await;
    }
    worker_executor
}

fn three_message_command(request_number: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_number),
        model: "astronomical/test-worker".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "Shall I compare thee to a summer's day?".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: Some("Thou art more lovely and more temperate.".to_owned()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ChatMessage::User {
                content: "Continue the sonnet.".to_owned(),
                images: Vec::new(),
            },
        ],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
    }
}
