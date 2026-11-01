//! Integration contract for the expert-streaming observatory across requests.
//!
//! The worker pushes one `ExpertStreamingStats` event per finalized
//! generation: its byte counters are engine-load accumulations that keep
//! growing across replies, while its prompt/generated token counts describe
//! only the finished request. The daemon-side observatory must keep both
//! views honest at once (issue #1715).

use std::time::Duration;

use astronomical_daemon::{ChatGenerationExecutor, ChatGenerationStreamEvent, WorkerHealthStatus};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use tokio::time::{Instant, sleep, timeout};

use crate::support::daemon::launch_test_executor;

/// Two requests through the session-stats fixture: the second event's
/// engine-load counters must describe the whole session so far (larger than
/// the first event's), the per-reply token counts must reset, and the
/// daemon-accumulated session totals must be the sum across events.
#[tokio::test]
async fn should_accumulate_session_totals_across_requests() {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-daemon-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_executor = launch_test_executor(worker_executable_path)
        .await
        .expect("the worker-backed executor should launch the fixture worker");
    wait_for_health(&worker_executor, WorkerHealthStatus::Ready).await;

    complete_session_fixture_request(&worker_executor, 1).await;
    let first_snapshot = worker_executor.worker_health_snapshot();
    let first_stats = first_snapshot
        .expert_streaming_observatory
        .latest_stats
        .as_ref()
        .expect("the first stats event must reach the observatory");
    assert_eq!(first_stats.mandatory_prefill_source_bytes, 10_000_000_000);
    assert_eq!(first_stats.prompt_token_count, 100);
    assert_eq!(
        first_snapshot
            .expert_streaming_observatory
            .accumulated_prompt_token_count,
        100
    );

    complete_session_fixture_request(&worker_executor, 2).await;
    let second_snapshot = worker_executor.worker_health_snapshot();
    let second_stats = second_snapshot
        .expert_streaming_observatory
        .latest_stats
        .as_ref()
        .expect("the second stats event must reach the observatory");
    // Engine-load byte counters accumulate: the second event already includes
    // the first request's reads plus its own.
    assert_eq!(
        second_stats.mandatory_prefill_source_bytes, 20_000_000_000,
        "the worker's engine-load counters must keep accumulating across replies"
    );
    assert!(
        second_stats.mandatory_prefill_source_bytes > first_stats.mandatory_prefill_source_bytes
    );
    // Per-reply token counts reset with each event; the daemon accumulates.
    assert_eq!(second_stats.prompt_token_count, 100);
    assert_eq!(
        second_snapshot
            .expert_streaming_observatory
            .accumulated_prompt_token_count,
        200,
        "the daemon must accumulate prompt tokens across stats events"
    );
    assert_eq!(
        second_snapshot
            .expert_streaming_observatory
            .accumulated_generated_token_count,
        30
    );

    worker_executor
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

/// Runs one chat request against the session fixture and waits for its
/// completion event so the stats publication has been processed.
async fn complete_session_fixture_request(
    worker_executor: &astronomical_daemon::WorkerHandle,
    request_sequence: u64,
) {
    let mut stream_receiver = worker_executor
        .start_chat_generation(session_fixture_command(request_sequence))
        .await
        .expect("the session fixture should start the request");
    timeout(Duration::from_secs(5), async {
        loop {
            match stream_receiver
                .recv()
                .await
                .expect("the stream should deliver another event")
            {
                ChatGenerationStreamEvent::Completed { .. } => break,
                ChatGenerationStreamEvent::Failed { reason } => {
                    panic!("the session fixture request failed: {reason:?}")
                }
                _ => continue,
            }
        }
    })
    .await
    .expect("the session fixture request must complete");
    sleep(Duration::from_millis(100)).await;
}

fn session_fixture_command(request_sequence: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_sequence),
        model: String::from("astronomical/expert-streaming-session-fixture"),
        messages: vec![ChatMessage::User {
            content: "accumulate session totals".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
    }
}

async fn wait_for_health(
    worker_executor: &astronomical_daemon::WorkerHandle,
    expected_status: WorkerHealthStatus,
) {
    let health_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let health_snapshot = worker_executor.worker_health_snapshot();
        if health_snapshot.status == expected_status {
            return;
        }
        assert!(
            Instant::now() < health_deadline,
            "worker health did not reach {expected_status:?}; last status was {:?}",
            health_snapshot.status
        );
        sleep(Duration::from_millis(50)).await;
    }
}
