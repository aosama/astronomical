use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatImageInput,
    ChatMessage, ChatToolChoice, ExpertMemoryMode, MAX_IPC_FRAME_BYTES, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    GenerationStartError, MlxMemoryLimitUpdateOutcome, WorkerHandle, WorkerHealthSnapshot,
    WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 20480;

#[tokio::test]
async fn should_launch_without_a_literal_host_memory_envelope() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-test-worker")
            .expect("Cargo should provide the test worker fixture path"),
    );
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temp_dir.path())
        .expect("test performance log should be created");

    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        performance_log,
        Arc::new(HashMap::new()),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("machine-specific MLX limits belong to the inference worker");

    worker_handle
        .shutdown()
        .await
        .expect("the scripted worker should shut down");
}

#[tokio::test]
async fn should_load_the_requested_model_only_after_the_first_generation_request() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temp_dir.path())
        .expect("test performance log should be created");
    let requested_model_id = "astronomical/requested-model".to_owned();
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        performance_log,
        Arc::new(HashMap::from([(
            requested_model_id.clone(),
            PathBuf::from("/models/requested-model"),
        )])),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");

    wait_for_idle_worker(&worker_handle).await;
    assert!(
        worker_handle
            .worker_health_snapshot()
            .ready_model_id
            .is_none()
    );

    let mut generation_events = worker_handle
        .start_chat_generation(chat_command(requested_model_id))
        .await
        .expect("the first request should load the requested model");
    assert!(matches!(
        generation_events.recv().await,
        Some(ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        })
    ));
    assert_eq!(
        worker_handle
            .worker_health_snapshot()
            .ready_model_id
            .as_deref(),
        Some("astronomical/requested-model")
    );
    assert_eq!(
        worker_handle
            .worker_health_snapshot()
            .minimum_mlx_memory_ceiling_bytes,
        3_000_000_000
    );
    assert_eq!(
        worker_handle.worker_health_snapshot().expert_memory_mode,
        Some(ExpertMemoryMode::Resident),
        "health must publish the expert mode selected before the replacement model became ready"
    );
    worker_handle
        .shutdown()
        .await
        .expect("the idle worker should shut down");
}

#[tokio::test]
async fn should_apply_a_memory_limit_immediately_when_worker_is_idle() {
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
        Arc::new(HashMap::new()),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    assert_eq!(
        worker_handle
            .update_mlx_memory_limit(32_000_000_000)
            .await
            .expect("the idle memory limit should apply"),
        MlxMemoryLimitUpdateOutcome::Applied
    );
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(
        worker_health_snapshot.mlx_memory_ceiling_bytes,
        32_000_000_000
    );
    assert_eq!(
        worker_health_snapshot.pending_mlx_memory_ceiling_bytes,
        None
    );
    assert_eq!(worker_health_snapshot.mlx_memory_limit_error, None);
    assert_eq!(worker_health_snapshot.ready_model_id, None);
    assert_eq!(worker_health_snapshot.expert_memory_mode, None);
    worker_handle
        .shutdown()
        .await
        .expect("the idle worker should shut down");
}

#[tokio::test]
async fn should_contain_a_worker_that_does_not_acknowledge_a_memory_limit_update() {
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
        Arc::new(HashMap::new()),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    let update_error = timeout(
        Duration::from_secs(2),
        worker_handle.update_mlx_memory_limit(30_000_000_000),
    )
    .await
    .expect("the supervisor should bound the acknowledgement wait")
    .expect_err("an unacknowledged memory update should fail");

    assert!(
        update_error
            .to_string()
            .contains("memory-limit update acknowledgement")
    );
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Unavailable
    );
    worker_handle
        .shutdown()
        .await
        .expect("the contained worker should already be reaped");
}

#[tokio::test]
async fn should_remain_available_after_the_first_requested_model_fails_to_load() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temp_dir.path())
        .expect("test performance log should be created");
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        performance_log,
        Arc::new(HashMap::from([
            (
                "astronomical/invalid-model".to_owned(),
                PathBuf::from("/models/invalid-model"),
            ),
            (
                "astronomical/requested-model".to_owned(),
                PathBuf::from("/models/requested-model"),
            ),
        ])),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    let invalid_model_start = worker_handle
        .start_chat_generation(chat_command("astronomical/invalid-model".to_owned()))
        .await;
    assert_eq!(
        invalid_model_start.expect_err("the invalid model should fail before generation"),
        GenerationStartError::ModelLoadFailed {
            model_load_failure_reason: "model artifact validation failed: OptiQ metadata uses unsupported 2-bit quantization".to_owned(),
        }
    );
    assert_eq!(
        worker_handle.worker_health_snapshot(),
        WorkerHealthSnapshot::ready_without_model(40_000_000_000)
    );

    let mut generation_events = worker_handle
        .start_chat_generation(chat_command("astronomical/requested-model".to_owned()))
        .await
        .expect("the worker should load a valid model after the rejected model");
    assert!(matches!(
        generation_events.recv().await,
        Some(ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        })
    ));
    worker_handle
        .shutdown()
        .await
        .expect("the recovered idle worker should shut down");
}

#[tokio::test]
async fn should_reject_an_oversized_generation_command_without_terminating_the_loaded_worker() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_log_directory =
        tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temporary_log_directory.path())
        .expect("test performance log should be created");
    let requested_model_id = "astronomical/requested-model";
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        performance_log,
        Arc::new(HashMap::from([(
            requested_model_id.to_owned(),
            PathBuf::from("/models/requested-model"),
        )])),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    let mut oversized_generation_command = chat_command(requested_model_id.to_owned());
    oversized_generation_command.messages = vec![ChatMessage::User {
        content: "Describe this image.".to_owned(),
        images: vec![ChatImageInput {
            mime_type: "image/png".to_owned(),
            decoded_bytes: vec![0; MAX_IPC_FRAME_BYTES * 3 / 4],
        }],
    }];

    let oversized_generation_outcome = worker_handle
        .start_chat_generation(oversized_generation_command)
        .await;

    assert!(matches!(
        oversized_generation_outcome,
        Err(GenerationStartError::RequestTooLarge {
            actual_ipc_message_bytes,
            maximum_ipc_message_bytes: MAX_IPC_FRAME_BYTES,
        }) if actual_ipc_message_bytes > MAX_IPC_FRAME_BYTES
    ));
    let mut followup_generation_events = worker_handle
        .start_chat_generation(chat_command(requested_model_id.to_owned()))
        .await
        .expect("the loaded worker should remain available after rejecting the oversized request");
    assert!(matches!(
        followup_generation_events.recv().await,
        Some(ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        })
    ));
    worker_handle
        .shutdown()
        .await
        .expect("the reusable worker should shut down");
}

#[tokio::test]
async fn should_reject_an_unmapped_model_without_forwarding_generation_to_the_idle_worker() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temp_dir.path())
        .expect("test performance log should be created");
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(1),
        performance_log,
        Arc::new(HashMap::new()),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    let unknown_model_start = worker_handle
        .start_chat_generation(chat_command("astronomical/unknown-model".to_owned()))
        .await;

    assert!(matches!(
        unknown_model_start,
        Err(GenerationStartError::WorkerUnavailable)
    ));
    assert_eq!(
        worker_handle.worker_health_snapshot(),
        WorkerHealthSnapshot::ready_without_model(40_000_000_000)
    );
    worker_handle
        .shutdown()
        .await
        .expect("the idle worker should shut down after rejecting an unknown model");
}

#[tokio::test]
async fn should_bound_the_time_waiting_for_a_requested_model_to_load() {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temp_dir = tempfile::tempdir().expect("test performance log directory should be created");
    let performance_log = GenerationPerformanceLog::open(temp_dir.path())
        .expect("test performance log should be created");
    let worker_handle = WorkerHandle::launch(
        worker_executable_path,
        Duration::from_secs(2),
        performance_log,
        Arc::new(HashMap::from([(
            "astronomical/hanging-model".to_owned(),
            PathBuf::from("/models/hanging-model"),
        )])),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    .await
    .expect("the idle worker should launch");
    wait_for_idle_worker(&worker_handle).await;

    let hanging_model_start = timeout(
        Duration::from_secs(3),
        worker_handle.start_chat_generation(chat_command("astronomical/hanging-model".to_owned())),
    )
    .await;

    assert!(matches!(
        hanging_model_start,
        Ok(Err(GenerationStartError::WorkerUnavailable))
    ));
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Unavailable
    );
}

async fn wait_for_idle_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "idle worker did not become ready; last status was {:?}",
            worker_health_snapshot.status
        );
        sleep(Duration::from_millis(10)).await;
    }
}

fn chat_command(requested_model_id: String) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: requested_model_id,
        messages: vec![ChatMessage::User {
            content: "hello".to_owned(),
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
