use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, MlxMemoryLimitUpdateOutcome, WorkerActivity,
    WorkerHealthStatus,
};
use tokio::time::{Instant, interval, sleep, timeout};

use super::model_prefill_qualification_worker::{
    PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS, PREFILL_QUALIFICATION_MODEL_ID,
    build_prefill_qualification_prompt, configured_prefill_qualification_model_directory,
    launch_prepared_prefill_qualification_worker, prepare_prefill_qualification_worker,
    wait_until_prefill_qualification_worker_is_idle,
};

const PREFILL_CANCELLATION_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT: Duration = Duration::from_secs(60);
const CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS: u32 = 8_192;
const CAPACITY_RETRY_CALIBRATION_PREFILL_CHUNCK_TOKENS: u32 =
    CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS / 2;
const CAPACITY_RETRY_PROMPT_TOKEN_COUNT: usize = CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS as usize + 1;
const CAPACITY_RETRY_CALIBRATION_PROMPT_TOKEN_COUNT: usize =
    CAPACITY_RETRY_CALIBRATION_PREFILL_CHUNCK_TOKENS as usize + 1;
const FOLLOWUP_MAXIMUM_OUTPUT_TOKENS: u16 = 1;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calibrates a runtime-derived prefill capacity retry, then verifies cancellation recovery"]
async fn should_keep_the_worker_reusable_after_dynamic_prefill_capacity_retry_and_cancellation() {
    timeout(
        PREFILL_CANCELLATION_QUALIFICATION_TIMEOUT,
        run_prefill_cancellation_qualification(),
    )
    .await
    .expect("the prefill cancellation qualification must finish within 115 seconds");
}

async fn run_prefill_cancellation_qualification() {
    let qualification_label = "prefill-retry-cancellation";
    let configured_model_directory = configured_prefill_qualification_model_directory();
    let calibration_prompt_content = build_prefill_qualification_prompt(
        &configured_model_directory,
        CAPACITY_RETRY_CALIBRATION_PROMPT_TOKEN_COUNT,
    );
    let retry_prompt_content = build_prefill_qualification_prompt(
        &configured_model_directory,
        CAPACITY_RETRY_PROMPT_TOKEN_COUNT,
    );
    let prepared_prefill_qualification_worker = prepare_prefill_qualification_worker(
        &configured_model_directory,
        Some(CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS),
        None,
    );
    let worker_handle = launch_prepared_prefill_qualification_worker(
        &prepared_prefill_qualification_worker,
        &configured_model_directory,
    )
    .await;
    wait_until_prefill_qualification_worker_is_idle(&worker_handle, qualification_label).await;

    eprintln!(
        "[prefill-retry-cancellation] status=progress phase=calibration completed_chunk_tokens={CAPACITY_RETRY_CALIBRATION_PREFILL_CHUNCK_TOKENS} prompt_tokens={CAPACITY_RETRY_CALIBRATION_PROMPT_TOKEN_COUNT}"
    );
    let mut calibration_stream_receiver = worker_handle
        .start_chat_generation(generation_command(
            RequestId::new(50),
            calibration_prompt_content,
            PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the calibration prefill request should start");
    let (calibration_active_memory_bytes, calibration_peak_memory_bytes) =
        wait_for_full_calibration_prefill(
            &mut calibration_stream_receiver,
            CAPACITY_RETRY_CALIBRATION_PROMPT_TOKEN_COUNT,
        )
        .await;
    drop(calibration_stream_receiver);
    wait_for_worker_recovery_after_cancellation(&worker_handle).await;
    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    let post_calibration_stable_memory_bytes = worker_health_snapshot
        .latest_mlx_memory_snapshot
        .expect("the calibration cancellation should publish finalized MLX memory")
        .active_memory_bytes;
    let constrained_mlx_memory_ceiling_bytes = capacity_retry_memory_ceiling_bytes(
        post_calibration_stable_memory_bytes,
        calibration_active_memory_bytes,
        calibration_peak_memory_bytes,
        worker_health_snapshot.minimum_mlx_memory_ceiling_bytes,
        worker_health_snapshot.mlx_memory_ceiling_bytes,
    );
    assert_eq!(
        worker_handle
            .update_mlx_memory_limit(constrained_mlx_memory_ceiling_bytes)
            .await
            .expect("the idle worker should accept the runtime-derived memory ceiling"),
        MlxMemoryLimitUpdateOutcome::Applied,
        "the runtime-derived memory ceiling must apply before the retry request"
    );
    eprintln!(
        "[prefill-retry-cancellation] status=progress phase=retry requested_chunk_tokens={CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS} ceiling_bytes={constrained_mlx_memory_ceiling_bytes}"
    );
    let mut stream_receiver = worker_handle
        .start_chat_generation(generation_command(
            RequestId::new(51),
            retry_prompt_content,
            PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the retried prefill request should start");
    wait_for_reduced_prefill_chunck_after_capacity_retry(
        &mut stream_receiver,
        constrained_mlx_memory_ceiling_bytes,
    )
    .await;
    let cancellation_requested_at = Instant::now();
    drop(stream_receiver);
    wait_for_worker_recovery_after_cancellation(&worker_handle).await;
    let finalized_worker_health_snapshot = worker_handle.worker_health_snapshot();
    let finalized_mlx_memory_snapshot = finalized_worker_health_snapshot
        .latest_mlx_memory_snapshot
        .expect("the cancelled retry request should publish finalized MLX memory");
    assert!(
        finalized_mlx_memory_snapshot.active_memory_bytes <= constrained_mlx_memory_ceiling_bytes,
        "the cancelled retry request left stable memory above C: active_bytes={}, ceiling_bytes={constrained_mlx_memory_ceiling_bytes}",
        finalized_mlx_memory_snapshot.active_memory_bytes,
    );
    eprintln!(
        "[prefill-cancellation] cancellation_acknowledged_millis={}",
        cancellation_requested_at.elapsed().as_millis()
    );

    let mut followup_stream_receiver = worker_handle
        .start_chat_generation(generation_command(
            RequestId::new(52),
            "Reply with one word.".to_owned(),
            FOLLOWUP_MAXIMUM_OUTPUT_TOKENS,
        ))
        .await
        .expect("the recovered worker should accept a follow-up request");
    let followup_completion = timeout(
        CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT,
        receive_followup_completion(&mut followup_stream_receiver),
    )
    .await
    .expect("the recovered worker should complete a short follow-up request");
    assert_eq!(followup_completion, FOLLOWUP_MAXIMUM_OUTPUT_TOKENS);
    worker_handle
        .shutdown()
        .await
        .expect("the recovered worker should shut down cleanly");
}

async fn wait_for_full_calibration_prefill(
    calibration_stream_receiver: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    expected_prompt_token_count: usize,
) -> (u64, u64) {
    let calibration_started_at = Instant::now();
    let mut calibration_progress_interval = interval(Duration::from_secs(5));
    calibration_progress_interval.tick().await;
    loop {
        tokio::select! {
            calibration_stream_event = calibration_stream_receiver.recv() => {
                let calibration_stream_event = calibration_stream_event
                    .expect("the calibration stream should remain open through one full prefill chunk");
                match calibration_stream_event {
                    ChatGenerationStreamEvent::PrefillProgress {
                        processed_tokens,
                        total_tokens,
                        completed_prefill_chunck_tokens: Some(completed_prefill_chunck_tokens),
                        mlx_active_memory_bytes,
                        mlx_peak_memory_bytes,
                        ..
                    } if completed_prefill_chunck_tokens == CAPACITY_RETRY_CALIBRATION_PREFILL_CHUNCK_TOKENS => {
                        assert_eq!(total_tokens, expected_prompt_token_count as u32);
                        assert_eq!(
                            processed_tokens,
                            CAPACITY_RETRY_CALIBRATION_PREFILL_CHUNCK_TOKENS
                        );
                        let calibration_active_memory_bytes = mlx_active_memory_bytes
                            .expect("the calibration prefill should report active MLX memory");
                        let calibration_peak_memory_bytes = mlx_peak_memory_bytes
                            .expect("the calibration prefill should report peak MLX memory");
                        eprintln!(
                            "[prefill-retry-cancellation] status=progress phase=calibrated active_bytes={calibration_active_memory_bytes} peak_bytes={calibration_peak_memory_bytes}"
                        );
                        return (calibration_active_memory_bytes, calibration_peak_memory_bytes);
                    }
                    ChatGenerationStreamEvent::PrefillProgress { processed_tokens, total_tokens, .. } => {
                        eprintln!(
                            "[prefill-retry-cancellation] status=progress phase=calibration processed_tokens={processed_tokens}/{total_tokens}"
                        );
                    }
                    ChatGenerationStreamEvent::ReasoningFragment(_)
                    | ChatGenerationStreamEvent::TextFragment(_)
                    | ChatGenerationStreamEvent::ToolCall { .. }
                    | ChatGenerationStreamEvent::Completed { .. } => {
                        panic!("the calibration request reached decode before one full prefill chunk completed");
                    }
                    ChatGenerationStreamEvent::Failed { reason } => {
                        panic!("the calibration request failed before one full prefill chunk: {reason:?}");
                    }
                    ChatGenerationStreamEvent::Error(error_code) => {
                        panic!("the calibration stream failed before one full prefill chunk: {error_code:?}");
                    }
                }
            }
            _ = calibration_progress_interval.tick() => {
                eprintln!(
                    "[prefill-retry-cancellation] status=progress phase=calibration elapsed_seconds={}",
                    calibration_started_at.elapsed().as_secs()
                );
            }
        }
    }
}

fn capacity_retry_memory_ceiling_bytes(
    post_calibration_stable_memory_bytes: u64,
    calibration_active_memory_bytes: u64,
    calibration_peak_memory_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    current_mlx_memory_ceiling_bytes: u64,
) -> u64 {
    assert!(
        calibration_peak_memory_bytes > calibration_active_memory_bytes,
        "the calibration prefill must expose transient MLX memory: active_bytes={calibration_active_memory_bytes}, peak_bytes={calibration_peak_memory_bytes}"
    );
    let reclaimable_idle_residency_bytes =
        post_calibration_stable_memory_bytes.saturating_sub(minimum_mlx_memory_ceiling_bytes);
    assert!(
        reclaimable_idle_residency_bytes > 0,
        "the retry qualification needs some reclaimable expert residency"
    );
    let measured_retry_ceiling_bytes =
        minimum_mlx_memory_ceiling_bytes.saturating_add(reclaimable_idle_residency_bytes / 3);
    let constrained_mlx_memory_ceiling_bytes =
        measured_retry_ceiling_bytes.min(current_mlx_memory_ceiling_bytes.saturating_sub(1));
    assert!(
        constrained_mlx_memory_ceiling_bytes >= minimum_mlx_memory_ceiling_bytes,
        "the loaded model has no lower valid ceiling for a retry qualification: minimum_bytes={minimum_mlx_memory_ceiling_bytes}, current_bytes={current_mlx_memory_ceiling_bytes}"
    );
    assert!(
        allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes)
            < calibration_peak_memory_bytes,
        "the runtime-derived retry ceiling must reject the calibrated peak: ceiling_bytes={constrained_mlx_memory_ceiling_bytes}, allowed_peak_bytes={}, calibration_peak_bytes={calibration_peak_memory_bytes}",
        allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes),
    );
    constrained_mlx_memory_ceiling_bytes
}

async fn wait_for_reduced_prefill_chunck_after_capacity_retry(
    retry_stream_receiver: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
    constrained_mlx_memory_ceiling_bytes: u64,
) {
    let retry_started_at = Instant::now();
    let allowed_peak_memory_bytes = allowed_peak_memory_bytes(constrained_mlx_memory_ceiling_bytes);
    let mut retry_progress_interval = interval(Duration::from_secs(5));
    retry_progress_interval.tick().await;
    loop {
        tokio::select! {
            retry_stream_event = retry_stream_receiver.recv() => {
                let retry_stream_event = retry_stream_event
                    .expect("the retry stream should remain open through its first reduced prefill chunk");
                match retry_stream_event {
                    ChatGenerationStreamEvent::PrefillProgress {
                        processed_tokens,
                        total_tokens,
                        completed_prefill_chunck_tokens: Some(completed_prefill_chunck_tokens),
                        mlx_active_memory_bytes,
                        mlx_peak_memory_bytes,
                        ..
                    } => {
                        assert_eq!(total_tokens, CAPACITY_RETRY_PROMPT_TOKEN_COUNT as u32);
                        assert_eq!(processed_tokens, completed_prefill_chunck_tokens);
                        assert!(
                            completed_prefill_chunck_tokens < CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS,
                            "the first successful prefill chunk must be reduced after native capacity retry: requested_tokens={CAPACITY_RETRY_PREFILL_CHUNCK_TOKENS}, completed_tokens={completed_prefill_chunck_tokens}"
                        );
                        let observed_active_memory_bytes = mlx_active_memory_bytes
                            .expect("the retried prefill should report active MLX memory");
                        let observed_peak_memory_bytes = mlx_peak_memory_bytes
                            .expect("the retried prefill should report peak MLX memory");
                        assert!(
                            observed_active_memory_bytes <= constrained_mlx_memory_ceiling_bytes,
                            "the reduced prefill left stable memory above C: active_bytes={observed_active_memory_bytes}, ceiling_bytes={constrained_mlx_memory_ceiling_bytes}"
                        );
                        assert!(
                            observed_peak_memory_bytes <= allowed_peak_memory_bytes,
                            "the reduced prefill exceeded P: peak_bytes={observed_peak_memory_bytes}, allowed_peak_bytes={allowed_peak_memory_bytes}"
                        );
                        eprintln!(
                            "[prefill-retry-cancellation] status=progress phase=retried completed_chunk_tokens={completed_prefill_chunck_tokens} active_bytes={observed_active_memory_bytes} peak_bytes={observed_peak_memory_bytes}"
                        );
                        return;
                    }
                    ChatGenerationStreamEvent::PrefillProgress { processed_tokens, total_tokens, .. } => {
                        eprintln!(
                            "[prefill-retry-cancellation] status=progress phase=retry processed_tokens={processed_tokens}/{total_tokens}"
                        );
                    }
                    ChatGenerationStreamEvent::ReasoningFragment(_)
                    | ChatGenerationStreamEvent::TextFragment(_)
                    | ChatGenerationStreamEvent::ToolCall { .. }
                    | ChatGenerationStreamEvent::Completed { .. } => {
                        panic!("the retried request reached decode before a reduced prefill chunk completed");
                    }
                    ChatGenerationStreamEvent::Failed { reason } => {
                        panic!("the retried request failed before a reduced prefill chunk completed: {reason:?}");
                    }
                    ChatGenerationStreamEvent::Error(error_code) => {
                        panic!("the retry stream failed before a reduced prefill chunk completed: {error_code:?}");
                    }
                }
            }
            _ = retry_progress_interval.tick() => {
                eprintln!(
                    "[prefill-retry-cancellation] status=progress phase=retry elapsed_seconds={}",
                    retry_started_at.elapsed().as_secs()
                );
            }
        }
    }
}

const fn allowed_peak_memory_bytes(mlx_memory_ceiling_bytes: u64) -> u64 {
    mlx_memory_ceiling_bytes.saturating_add(mlx_memory_ceiling_bytes / 100)
}

async fn wait_for_worker_recovery_after_cancellation(
    worker_handle: &astronomical_supervisor::WorkerHandle,
) {
    let recovery_deadline = Instant::now() + CANCELLATION_ACKNOWLEDGEMENT_TIMEOUT;
    loop {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.activity == WorkerActivity::Idle
            && worker_health_snapshot.ready_model_id.as_deref()
                == Some(PREFILL_QUALIFICATION_MODEL_ID)
        {
            return;
        }
        assert!(
            Instant::now() < recovery_deadline,
            "the worker did not recover from prefill cancellation: {worker_health_snapshot:?}"
        );
        sleep(Duration::from_millis(100)).await;
    }
}

async fn receive_followup_completion(
    followup_stream_receiver: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
) -> u16 {
    while let Some(followup_stream_event) = followup_stream_receiver.recv().await {
        match followup_stream_event {
            ChatGenerationStreamEvent::Completed {
                generated_token_count,
                ..
            } => return generated_token_count,
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the follow-up request failed after cancellation: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the follow-up request stream failed after cancellation: {error_code:?}");
            }
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. }
            | ChatGenerationStreamEvent::PrefillProgress { .. } => {}
        }
    }
    panic!("the follow-up request stream closed before completion");
}

fn generation_command(
    request_id: RequestId,
    user_prompt_content: String,
    maximum_output_tokens: u16,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id,
        model: PREFILL_QUALIFICATION_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: user_prompt_content,
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: maximum_output_tokens,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
    }
}
