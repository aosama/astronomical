//! Component journeys for the relaunch that follows worker failure
//! containment: one crash recovers, a relaunch that never becomes ready stays
//! unavailable, and a crash loop stops relaunching.

use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, ChatGenerationStreamErrorCode, ChatGenerationStreamEvent,
    GenerationStartError, WorkerControlError, WorkerHandle, WorkerHealthStatus,
};
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use tokio::time::{Instant, sleep, timeout};

use crate::support::daemon;

/// The loop waits one second before each relaunch; a relaunch of the
/// loading-forever fixture then fails after this model-load timeout.
const LOADING_FOREVER_MODEL_LOAD_TIMEOUT: Duration = Duration::from_millis(40);
/// Three admitted relaunches of the loading-forever fixture, each one backoff
/// plus one model-load timeout, with slack for a loaded CI runner.
const CRASH_LOOP_BUDGET_SPENT_AFTER: Duration = Duration::from_secs(5);
/// Long enough to cover several backoffs, so a fourth relaunch would surface
/// as a `Loading` sample.
const POST_BUDGET_OBSERVATION: Duration = Duration::from_millis(2_500);

#[tokio::test]
async fn should_serve_the_next_chat_request_after_one_worker_crash() {
    let worker_executor = launch_scripted_worker().await;
    let mut crashed_stream = worker_executor
        .start_chat_generation(command_for_model(
            "astronomical/exit-after-chat-admission-fixture",
        ))
        .await
        .expect("the fixture should admit the request it exits on");
    assert_eq!(
        receive_event(&mut crashed_stream).await,
        ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
    );

    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Ready,
        Duration::from_secs(5),
    )
    .await;
    let mut followup_stream = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await
        .expect("the replacement worker should admit the next request");

    assert!(matches!(
        receive_event(&mut followup_stream).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));
    worker_executor
        .shutdown()
        .await
        .expect("the replacement worker should shut down");
}

/// The scripted worker exits when asked to adopt exactly this many bytes, the
/// way a real worker does on a fatal engine error during a ceiling change.
const SCRIPTED_WORKER_EXIT_ON_MLX_MEMORY_LIMIT_BYTES: u64 = 13;

#[tokio::test]
async fn should_relaunch_and_serve_after_the_worker_dies_during_a_memory_limit_change() {
    let worker_executor = launch_scripted_worker().await;

    let memory_limit_error = worker_executor
        .update_mlx_memory_limit(
            SCRIPTED_WORKER_EXIT_ON_MLX_MEMORY_LIMIT_BYTES,
            "memory-change-that-kills-the-worker".to_owned(),
        )
        .await
        .expect_err("a worker that exits mid change must fail the update");
    assert!(
        matches!(
            memory_limit_error,
            WorkerControlError::ContainedMlxMemoryLimitFailure { .. }
        ),
        "the caller must learn the worker was contained, got {memory_limit_error}"
    );

    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Ready,
        Duration::from_secs(5),
    )
    .await;
    let mut followup_stream = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await
        .expect("the relaunched worker should admit the next request");
    assert!(matches!(
        receive_event(&mut followup_stream).await,
        ChatGenerationStreamEvent::Completed { .. }
    ));
    worker_executor
        .shutdown()
        .await
        .expect("the relaunched worker should shut down");
}

#[tokio::test]
async fn should_stay_unavailable_when_the_relaunched_worker_never_becomes_ready() {
    let worker_executor = launch_loading_forever_worker().await;
    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Unavailable,
        Duration::from_secs(2),
    )
    .await;
    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Loading,
        Duration::from_secs(3),
    )
    .await;

    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Unavailable,
        Duration::from_secs(2),
    )
    .await;
    let rejected_start = worker_executor
        .start_chat_generation(command_for_model("astronomical/test-worker"))
        .await;

    assert!(matches!(
        rejected_start,
        Err(GenerationStartError::WorkerUnavailable)
    ));
    worker_executor
        .shutdown()
        .await
        .expect("the contained worker should already be reaped");
}

#[tokio::test]
async fn should_stop_relaunching_once_the_crash_loop_budget_is_spent() {
    let worker_executor = launch_loading_forever_worker().await;
    sleep(CRASH_LOOP_BUDGET_SPENT_AFTER).await;

    let observed_statuses = sample_health_statuses(&worker_executor, POST_BUDGET_OBSERVATION).await;

    assert!(
        !observed_statuses.contains(&WorkerHealthStatus::Loading)
            && !observed_statuses.contains(&WorkerHealthStatus::Ready),
        "a spent crash-loop budget must leave the worker unavailable without further \
         relaunches, observed {observed_statuses:?}"
    );
    worker_executor
        .shutdown()
        .await
        .expect("the contained worker should already be reaped");
}

async fn launch_scripted_worker() -> WorkerHandle {
    let worker_executable_path = std::env::var("CARGO_BIN_EXE_astronomical-daemon-test-worker")
        .expect("Cargo should provide the test worker path");
    let worker_executor = daemon::launch_test_executor(worker_executable_path)
        .await
        .expect("the worker should launch");
    wait_for_health(
        &worker_executor,
        WorkerHealthStatus::Ready,
        Duration::from_secs(5),
    )
    .await;
    worker_executor
}

async fn launch_loading_forever_worker() -> WorkerHandle {
    let worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-daemon-loading-forever-worker")
            .expect("Cargo should provide the loading-forever worker fixture path");
    daemon::launch_test_worker_with_model_load_timeout(
        worker_executable_path,
        LOADING_FOREVER_MODEL_LOAD_TIMEOUT,
    )
    .await
    .expect("the loading-forever worker should start")
}

async fn sample_health_statuses(
    worker_executor: &WorkerHandle,
    observation_period: Duration,
) -> Vec<WorkerHealthStatus> {
    let observation_deadline = Instant::now() + observation_period;
    let mut observed_statuses = Vec::new();
    while Instant::now() < observation_deadline {
        observed_statuses.push(worker_executor.worker_health_snapshot().status);
        sleep(Duration::from_millis(10)).await;
    }
    observed_statuses
}

async fn wait_for_health(
    worker_executor: &WorkerHandle,
    expected_status: WorkerHealthStatus,
    maximum_wait: Duration,
) {
    let health_deadline = Instant::now() + maximum_wait;
    loop {
        let observed_status = worker_executor.worker_health_snapshot().status;
        if observed_status == expected_status {
            return;
        }
        assert!(
            Instant::now() < health_deadline,
            "worker health remained {observed_status:?} while waiting for {expected_status:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
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
            repetition_penalty_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        structured_generation: None,
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
