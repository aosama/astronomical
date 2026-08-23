//! Worker-boundary acceptance coverage for shared admission and image finalization.

use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice,
    ImageGenerationCommand, ImageGenerationSettings, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, GenerationQueueDepth, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationExecutor, ImageGenerationTimeouts,
    MlxMemoryLimitUpdateOutcome, PromptCacheClearOutcome, WorkerControlError, WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

#[tokio::test]
async fn should_share_fifo_order_and_capacity_between_chat_and_image_requests() {
    let worker_handle = launch_scripted_worker().await;
    let active_chat = worker_handle
        .start_chat_generation(chat_command(
            "astronomical/delayed-fragment-chat-fixture",
            1,
        ))
        .await
        .expect("the first chat request should become active");
    let queued_worker_handle = worker_handle.clone();
    let queued_image_task = tokio::spawn(async move {
        queued_worker_handle
            .start_image_generation(image_command(100, "completed-image"))
            .await
    });
    sleep(Duration::from_millis(50)).await;
    assert!(!queued_image_task.is_finished());

    let mut queued_chat_tasks = Vec::new();
    for queue_position in 1..GenerationQueueDepth::value() {
        let queued_worker_handle = worker_handle.clone();
        queued_chat_tasks.push(tokio::spawn(async move {
            queued_worker_handle
                .start_chat_generation(chat_command(
                    "astronomical/test-worker",
                    u64::try_from(queue_position).unwrap_or(u64::MAX) + 1,
                ))
                .await
        }));
    }
    sleep(Duration::from_millis(50)).await;
    assert!(matches!(
        worker_handle
            .start_image_generation(image_command(110, "completed-image"))
            .await,
        Err(GenerationStartError::CapacityUnavailable)
    ));

    drop(active_chat);
    let mut image_receiver = timeout(Duration::from_secs(2), queued_image_task)
        .await
        .expect("the first queued modality should start")
        .expect("the queued image task should not panic")
        .expect("the queued image should be admitted");
    assert!(
        image_receiver
            .recv()
            .await
            .expect("image outcome should arrive")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_cancel_a_disconnected_image_and_reuse_the_worker() {
    let worker_handle = launch_scripted_worker().await;
    let image_receiver = worker_handle
        .start_image_generation(image_command(100, "delayed-image-generation-fixture"))
        .await
        .expect("the delayed image should start");
    drop(image_receiver);

    let mut followup_receiver = timeout(
        Duration::from_secs(2),
        worker_handle.start_image_generation(image_command(101, "completed-image")),
    )
    .await
    .expect("disconnect cancellation should remain bounded")
    .expect("the worker should accept a follow-up image");
    assert!(
        timeout(Duration::from_secs(1), followup_receiver.recv())
            .await
            .expect("disconnect follow-up should remain bounded")
            .expect("follow-up should finish")
            .is_ok()
    );
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_release_a_failed_image_only_after_finalization_and_reuse_the_worker() {
    let worker_handle = launch_scripted_worker().await;
    let mut failed_receiver = worker_handle
        .start_image_generation(image_command(102, "failed-image-generation-fixture"))
        .await
        .expect("the failed image should start");
    assert!(
        failed_receiver
            .recv()
            .await
            .expect("failure should arrive")
            .is_err()
    );

    let mut followup_receiver = worker_handle
        .start_image_generation(image_command(103, "completed-image"))
        .await
        .expect("the finalized failure should release ownership");
    assert!(
        followup_receiver
            .recv()
            .await
            .expect("follow-up should finish")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_hold_completed_image_bytes_until_cleanup_finalization() {
    let worker_handle = launch_scripted_worker().await;
    let mut image_receiver = worker_handle
        .start_image_generation(image_command(104, "completion-before-finalization-fixture"))
        .await
        .expect("the image should start");

    assert!(
        timeout(Duration::from_millis(50), image_receiver.recv())
            .await
            .is_err()
    );
    assert!(
        timeout(Duration::from_secs(1), image_receiver.recv())
            .await
            .expect("finalization should release the result")
            .expect("the result channel should remain open")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

#[tokio::test]
async fn should_attribute_image_performance_only_after_finalization() {
    let executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let performance_log_directory =
        tempfile::tempdir().expect("the performance log directory should be created");
    let worker_handle =
        crate::common::supervisor::launch_test_executor_with_performance_log_directory(
            executable_path,
            performance_log_directory.path(),
        )
        .await
        .expect("the scripted worker should launch");
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;

    let mut image_receiver = worker_handle
        .start_image_generation(image_command(107, "completion-before-finalization-fixture"))
        .await
        .expect("the image should start");
    assert!(
        timeout(Duration::from_millis(50), image_receiver.recv())
            .await
            .is_err()
    );
    let log_before_finalization =
        std::fs::read_to_string(performance_log_directory.path().join("performance.jsonl"))
            .expect("the performance log should remain readable");
    assert!(log_before_finalization.is_empty());

    assert!(
        timeout(Duration::from_secs(1), image_receiver.recv())
            .await
            .expect("finalization should release the result")
            .expect("the result channel should remain open")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
    let image_performance_record: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(performance_log_directory.path().join("performance.jsonl"))
            .expect("the image performance record should be readable")
            .trim(),
    )
    .expect("the image performance record should be valid JSON");
    assert_eq!(image_performance_record["operation"], "image_generation");
    assert_eq!(image_performance_record["request_id"], 107);
    assert_eq!(image_performance_record["completion_outcome"], "completed");
    assert_eq!(
        image_performance_record["worker_reported_elapsed_millis"],
        30
    );
    assert!(image_performance_record["total_elapsed_millis"].is_u64());
    for attribution_field in [
        "queue_wait_elapsed_millis",
        "swap_load_elapsed_millis",
        "execution_elapsed_millis",
        "finalization_elapsed_millis",
    ] {
        assert!(image_performance_record[attribution_field].is_u64());
    }
    let attributed_elapsed_millis = image_performance_record["queue_wait_elapsed_millis"]
        .as_u64()
        .unwrap_or(u64::MAX)
        .saturating_add(
            image_performance_record["swap_load_elapsed_millis"]
                .as_u64()
                .unwrap_or(u64::MAX),
        )
        .saturating_add(
            image_performance_record["execution_elapsed_millis"]
                .as_u64()
                .unwrap_or(u64::MAX),
        )
        .saturating_add(
            image_performance_record["finalization_elapsed_millis"]
                .as_u64()
                .unwrap_or(u64::MAX),
        );
    assert!(
        image_performance_record["total_elapsed_millis"]
            .as_u64()
            .is_some_and(|total| total >= attributed_elapsed_millis)
    );
    assert!(
        image_performance_record["encoded_image_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
}

#[tokio::test]
async fn should_contain_an_image_request_without_a_bounded_cancellation_acknowledgement() {
    let executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_handle =
        crate::common::supervisor::launch_test_executor_with_cancellation_acknowledgement_timeout(
            executable_path,
            Duration::from_millis(100),
        )
        .await
        .expect("the scripted worker should launch");
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;
    let image_receiver = worker_handle
        .start_image_generation(image_command(
            105,
            "unacknowledged-image-cancellation-fixture",
        ))
        .await
        .expect("the unacknowledged image should start");
    drop(image_receiver);

    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;
    let mut followup_receiver = worker_handle
        .start_image_generation(image_command(108, "completed-image"))
        .await
        .expect("the replacement worker should accept a follow-up image");
    assert!(
        timeout(Duration::from_secs(1), followup_receiver.recv())
            .await
            .expect("replacement follow-up should remain bounded")
            .expect("follow-up should finish")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("replacement worker should shut down");
}

#[tokio::test]
async fn should_cancel_image_execution_and_progress_stalls_with_the_shared_bounded_path() {
    for (request_id, prompt, timeouts) in [
        (
            109,
            "delayed-image-generation-fixture",
            ImageGenerationTimeouts::new(Duration::from_secs(1), Duration::from_secs(3)),
        ),
        (
            110,
            "progress-stall-image-fixture",
            ImageGenerationTimeouts::new(Duration::from_secs(3), Duration::from_secs(1)),
        ),
        (
            118,
            "duplicate-progress-stall-image-fixture",
            ImageGenerationTimeouts::new(Duration::from_secs(3), Duration::from_secs(1)),
        ),
    ] {
        let executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
            .expect("Cargo should provide the scripted worker fixture path");
        let worker_handle =
            crate::common::supervisor::launch_test_executor_with_image_generation_timeouts(
                executable_path,
                Duration::from_millis(250),
                timeouts,
            )
            .await
            .expect("the scripted worker should launch");
        wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;
        let mut image_receiver = worker_handle
            .start_image_generation(image_command(request_id, prompt))
            .await
            .expect("the bounded image should start");
        assert!(matches!(
            timeout(Duration::from_secs(2), image_receiver.recv()).await,
            Ok(Some(Err(ImageGenerationExecutionError::DeadlineExceeded)))
        ));
        let mut followup_receiver = worker_handle
            .start_image_generation(image_command(request_id + 100, "completed-image"))
            .await
            .expect("deadline cancellation should preserve worker reuse");
        let followup_outcome = timeout(Duration::from_secs(2), followup_receiver.recv())
            .await
            .expect("deadline follow-up should remain bounded")
            .expect("follow-up should finish");
        assert!(
            followup_outcome.is_ok(),
            "deadline follow-up failed: {followup_outcome:?}"
        );
        worker_handle
            .shutdown()
            .await
            .expect("worker should shut down");
    }
}

#[tokio::test]
async fn should_refresh_the_stall_deadline_for_every_monotonic_image_progress_event() {
    let executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_handle =
        crate::common::supervisor::launch_test_executor_with_image_generation_timeouts(
            executable_path,
            Duration::from_millis(250),
            ImageGenerationTimeouts::new(Duration::from_secs(3), Duration::from_secs(1)),
        )
        .await
        .expect("the scripted worker should launch");
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;

    let mut image_receiver = worker_handle
        .start_image_generation(image_command(114, "elapsed-progress-refresh-image-fixture"))
        .await
        .expect("the progressing image should start");

    let image_outcome = timeout(Duration::from_secs(3), image_receiver.recv())
        .await
        .expect("monotonic progress should keep the request alive")
        .expect("the image outcome should arrive");
    assert!(
        image_outcome.is_ok(),
        "progressing image failed: {image_outcome:?}"
    );
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}

#[tokio::test]
async fn should_contain_an_image_completion_with_mismatched_guidance() {
    let worker_handle = launch_scripted_worker().await;
    let mut image_receiver = worker_handle
        .start_image_generation(image_command(115, "guidance-mismatch-image-fixture"))
        .await
        .expect("the malformed image completion should reach protocol validation");

    assert!(matches!(
        timeout(Duration::from_secs(1), image_receiver.recv()).await,
        Ok(Some(Err(ImageGenerationExecutionError::WorkerUnavailable))) | Ok(None)
    ));
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Unavailable).await;
    worker_handle
        .shutdown()
        .await
        .expect("contained shutdown should be idempotent");
}

#[tokio::test]
async fn should_contain_malformed_or_dimension_mismatched_png_completions() {
    for (request_id, prompt) in [
        (116, "malformed-png-image-fixture"),
        (117, "png-dimension-mismatch-image-fixture"),
    ] {
        let worker_handle = launch_scripted_worker().await;
        let mut image_receiver = worker_handle
            .start_image_generation(image_command(request_id, prompt))
            .await
            .expect("the invalid PNG completion should reach protocol validation");

        assert!(matches!(
            timeout(Duration::from_secs(1), image_receiver.recv()).await,
            Ok(Some(Err(ImageGenerationExecutionError::WorkerUnavailable))) | Ok(None)
        ));
        wait_for_worker_status(&worker_handle, WorkerHealthStatus::Unavailable).await;
        worker_handle
            .shutdown()
            .await
            .expect("contained shutdown should be idempotent");
    }
}

#[tokio::test]
async fn should_contain_every_non_monotonic_image_progress_dimension() {
    for (request_id, prompt) in [
        (111, "phase-regression-image-fixture"),
        (112, "step-regression-image-fixture"),
        (113, "elapsed-regression-image-fixture"),
    ] {
        let worker_handle = launch_scripted_worker().await;
        let mut image_receiver = worker_handle
            .start_image_generation(image_command(request_id, prompt))
            .await
            .expect("the malformed image request should reach protocol validation");
        assert!(matches!(
            timeout(Duration::from_secs(1), image_receiver.recv()).await,
            Ok(Some(Err(ImageGenerationExecutionError::WorkerUnavailable))) | Ok(None)
        ));
        wait_for_worker_status(&worker_handle, WorkerHealthStatus::Unavailable).await;
        worker_handle
            .shutdown()
            .await
            .expect("contained shutdown should be idempotent");
    }
}

#[tokio::test]
async fn should_keep_memory_cache_and_replacement_controls_busy_until_image_finalization() {
    let worker_handle = launch_scripted_worker().await;
    let image_receiver = worker_handle
        .start_image_generation(image_command(106, "delayed-image-generation-fixture"))
        .await
        .expect("the delayed image should start");

    assert_eq!(
        worker_handle
            .update_mlx_memory_limit(32_000_000_000, "queued-memory-generation".to_owned())
            .await
            .expect("memory update should queue"),
        MlxMemoryLimitUpdateOutcome::Queued,
    );
    assert_eq!(
        worker_handle
            .clear_prompt_cache(None)
            .await
            .expect("cache clear should queue"),
        PromptCacheClearOutcome::Queued,
    );
    assert!(matches!(
        worker_handle
            .restart_worker(
                std::path::PathBuf::from("unused-worker"),
                std::sync::Arc::new(std::collections::HashMap::new()),
                "unused-generation".to_owned(),
            )
            .await,
        Err(WorkerControlError::GenerationBusy)
    ));

    drop(image_receiver);
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;
    worker_handle
        .shutdown()
        .await
        .expect("shutdown should succeed");
}

async fn launch_scripted_worker() -> astronomical_supervisor::WorkerHandle {
    let executable_path = std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
        .expect("Cargo should provide the scripted worker fixture path");
    let worker_handle = crate::common::supervisor::launch_test_executor(executable_path)
        .await
        .expect("the scripted worker should launch");
    wait_for_worker_status(&worker_handle, WorkerHealthStatus::Ready).await;
    worker_handle
}

async fn wait_for_worker_status(
    worker_handle: &astronomical_supervisor::WorkerHandle,
    expected_status: WorkerHealthStatus,
) {
    let status_deadline = Instant::now() + Duration::from_secs(2);
    while worker_handle.worker_health_snapshot().status != expected_status {
        assert!(
            Instant::now() < status_deadline,
            "worker did not reach {expected_status:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

fn image_command(request_id: u64, prompt: &str) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: "astronomical/test-worker".to_owned(),
        prompt: prompt.to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 64,
            height_pixels: 64,
            steps: 4,
            guidance_thousandths: 1_000,
            seed: 7,
        },
    }
}

fn chat_command(model: &str, request_id: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: model.to_owned(),
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
