//! Acceptance coverage for queued generation across a lazy model swap.
//!
//! The success journey proves idle telemetry can be observed before the swap
//! acknowledgement without losing the queued request. The failure journey proves
//! that sharing the event handler does not weaken generation correlation rules.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatGenerationSettings, ChatMessage,
    ChatToolChoice, ImageGenerationCommand, ImageGenerationSettings, RequestId,
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerFlux2KleinModelConfiguration, WorkerImageGenerationModelFamily, WorkerLogLevel,
    WorkerModelConfiguration, WorkerStartupConfiguration,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    GenerationStartError, ImageGenerationExecutor, RuntimeModelPolicy, WorkerHandle,
    WorkerHealthStatus,
};
use tokio::time::{Instant, sleep, timeout};

const DELAYED_COMPLETION_MODEL_ID: &str = "astronomical/delayed-completion-model";
const GENERATION_EVENT_BEFORE_SWAP_MODEL_ID: &str =
    "astronomical/generation-event-before-swap-model";
pub(super) const TELEMETRY_BEFORE_SWAP_MODEL_ID: &str = "astronomical/telemetry-before-swap-model";
const IMAGE_MODEL_ID: &str = "astronomical/image-generation-model";
const INVALID_IMAGE_MODEL_ID: &str = "astronomical/invalid-image-generation-model";
const DELAYED_POLICY_ACK_MODEL_ID: &str = "astronomical/delayed-policy-ack-model";
const DELAYED_IMAGE_POLICY_ACK_MODEL_ID: &str = "astronomical/delayed-image-policy-ack-model";

#[tokio::test]
async fn should_complete_a_queued_model_swap_when_idle_telemetry_arrives_before_acknowledgement() {
    let worker_handle = launch_idle_worker_fixture().await;
    let mut first_generation_events = worker_handle
        .start_chat_generation(chat_command(DELAYED_COMPLETION_MODEL_ID, 1))
        .await
        .expect("the first model should load and start generation");

    let queued_worker_handle = worker_handle.clone();
    let queued_generation_task = tokio::spawn(async move {
        queued_worker_handle
            .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 2))
            .await
    });

    assert_generation_completed(&mut first_generation_events).await;
    let mut queued_generation_events = timeout(Duration::from_secs(2), queued_generation_task)
        .await
        .expect("the queued model swap should finish before the timeout")
        .expect("the queued generation task should not panic")
        .expect("idle telemetry must not make the queued model swap unavailable");
    assert_generation_completed(&mut queued_generation_events).await;

    let worker_health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(worker_health_snapshot.status, WorkerHealthStatus::Ready);
    assert_eq!(
        worker_health_snapshot.ready_model_id.as_deref(),
        Some(TELEMETRY_BEFORE_SWAP_MODEL_ID)
    );
    assert_eq!(
        worker_health_snapshot
            .ready_model_capabilities
            .as_ref()
            .expect("loaded model capabilities should be acknowledged")
            .chat
            .as_ref()
            .expect("the loaded model should advertise chat capabilities")
            .max_output_tokens,
        64
    );
    assert_eq!(
        worker_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .expect("loaded model policy should be acknowledged")
            .loaded_model
            .as_ref()
            .expect("the acknowledgement should identify the loaded model")
            .model_id(),
        TELEMETRY_BEFORE_SWAP_MODEL_ID
    );
    worker_handle
        .shutdown()
        .await
        .expect("the worker should remain available for graceful shutdown");
}

#[tokio::test]
async fn should_reject_a_generation_scoped_event_while_waiting_for_model_swap() {
    let worker_handle = launch_idle_worker_fixture().await;

    let generation_start_outcome = worker_handle
        .start_chat_generation(chat_command(GENERATION_EVENT_BEFORE_SWAP_MODEL_ID, 3))
        .await;

    assert!(matches!(
        generation_start_outcome,
        Err(GenerationStartError::WorkerUnavailable)
    ));
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
async fn should_swap_chat_to_image_to_chat_with_exact_runtime_policy() {
    let worker_handle = launch_idle_worker_fixture().await;
    let mut first_chat = worker_handle
        .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 10))
        .await
        .expect("the chat model should load");
    assert_generation_completed(&mut first_chat).await;

    let mut image_result = worker_handle
        .start_image_generation(ImageGenerationCommand {
            request_id: RequestId::new(11),
            model: IMAGE_MODEL_ID.to_owned(),
            prompt: "A moonlit balcony scene from Romeo and Juliet".to_owned(),
            settings: ImageGenerationSettings {
                width_pixels: 1_024,
                height_pixels: 1_024,
                steps: 4,
                guidance_thousandths: 1_000,
                seed: 7,
            },
        })
        .await
        .expect("the image model should load");
    assert!(
        image_result
            .recv()
            .await
            .expect("image result should arrive")
            .is_ok()
    );
    assert_eq!(
        worker_handle
            .worker_health_snapshot()
            .worker_runtime_feature_configuration
            .as_ref()
            .and_then(|configuration| configuration.loaded_model.as_ref())
            .map(|configuration| configuration.model_id()),
        Some(IMAGE_MODEL_ID),
    );

    let mut final_chat = worker_handle
        .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 12))
        .await
        .expect("the chat model should reload");
    assert_generation_completed(&mut final_chat).await;
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

#[tokio::test]
async fn should_recover_from_an_image_model_swap_failure_without_poisoning_the_worker() {
    let worker_handle = launch_idle_worker_fixture().await;
    let failed_start = worker_handle
        .start_image_generation(image_generation_command(INVALID_IMAGE_MODEL_ID, 13))
        .await;
    assert!(matches!(
        failed_start,
        Err(GenerationStartError::ModelLoadFailed { .. })
    ));

    let mut valid_image = worker_handle
        .start_image_generation(image_generation_command(IMAGE_MODEL_ID, 14))
        .await
        .expect("a valid image model should still load after rejection");
    assert!(
        valid_image
            .recv()
            .await
            .expect("image result should arrive")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("the worker should shut down");
}

#[tokio::test]
async fn should_publish_model_identity_and_runtime_policy_as_one_health_snapshot() {
    let worker_handle = launch_idle_worker_fixture().await;
    let mut first_generation = worker_handle
        .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 15))
        .await
        .expect("the initial model should load");
    assert_generation_completed(&mut first_generation).await;

    let swapping_worker_handle = worker_handle.clone();
    let swap_task = tokio::spawn(async move {
        swapping_worker_handle
            .start_chat_generation(chat_command(DELAYED_POLICY_ACK_MODEL_ID, 16))
            .await
    });
    sleep(Duration::from_millis(75)).await;
    let staged_health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(
        staged_health_snapshot.ready_model_id.as_deref(),
        Some(TELEMETRY_BEFORE_SWAP_MODEL_ID),
    );
    assert_eq!(
        staged_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .and_then(|configuration| configuration.loaded_model.as_ref())
            .map(|configuration| configuration.model_id()),
        Some(TELEMETRY_BEFORE_SWAP_MODEL_ID),
    );

    let mut swapped_generation = timeout(Duration::from_secs(1), swap_task)
        .await
        .expect("the exact swap acknowledgements should arrive")
        .expect("the swap task should not panic")
        .expect("the swapped generation should start");
    assert_generation_completed(&mut swapped_generation).await;
    let committed_health_snapshot = worker_handle.worker_health_snapshot();
    assert_eq!(
        committed_health_snapshot.ready_model_id.as_deref(),
        Some(DELAYED_POLICY_ACK_MODEL_ID)
    );
    assert_eq!(
        committed_health_snapshot
            .worker_runtime_feature_configuration
            .as_ref()
            .and_then(|configuration| configuration.loaded_model.as_ref())
            .map(|configuration| configuration.model_id()),
        Some(DELAYED_POLICY_ACK_MODEL_ID),
    );
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}

#[tokio::test]
async fn should_not_dispatch_an_image_after_disconnect_during_model_swap() {
    let worker_handle = launch_idle_worker_fixture().await;
    let disconnected_worker_handle = worker_handle.clone();
    let disconnected_request = tokio::spawn(async move {
        disconnected_worker_handle
            .start_image_generation(ImageGenerationCommand {
                request_id: RequestId::new(17),
                model: DELAYED_IMAGE_POLICY_ACK_MODEL_ID.to_owned(),
                prompt: "must-not-dispatch-after-disconnect".to_owned(),
                settings: ImageGenerationSettings {
                    width_pixels: 1_024,
                    height_pixels: 1_024,
                    steps: 4,
                    guidance_thousandths: 1_000,
                    seed: 7,
                },
            })
            .await
    });
    sleep(Duration::from_millis(75)).await;
    disconnected_request.abort();
    sleep(Duration::from_millis(250)).await;
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );

    let mut followup_image = worker_handle
        .start_image_generation(image_generation_command(
            DELAYED_IMAGE_POLICY_ACK_MODEL_ID,
            18,
        ))
        .await
        .expect("the completed swap should remain reusable");
    assert!(
        timeout(Duration::from_secs(1), followup_image.recv())
            .await
            .expect("follow-up image should remain bounded")
            .expect("follow-up should finish")
            .is_ok()
    );
    worker_handle
        .shutdown()
        .await
        .expect("worker should shut down");
}

pub(super) async fn launch_idle_worker_fixture() -> WorkerHandle {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-supervisor-idle-worker")
            .expect("Cargo should provide the idle worker fixture path"),
    );
    let temporary_log_directory =
        tempfile::tempdir().expect("test performance log directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        Duration::from_secs(1),
        GenerationPerformanceLog::open(temporary_log_directory.path())
            .expect("test performance log should be created"),
        Arc::new(HashMap::from([
            (
                DELAYED_COMPLETION_MODEL_ID.to_owned(),
                runtime_model_policy(
                    DELAYED_COMPLETION_MODEL_ID,
                    "/models/delayed-completion-model",
                    128,
                ),
            ),
            (
                TELEMETRY_BEFORE_SWAP_MODEL_ID.to_owned(),
                runtime_model_policy(
                    TELEMETRY_BEFORE_SWAP_MODEL_ID,
                    "/models/telemetry-before-swap-model",
                    64,
                ),
            ),
            (
                GENERATION_EVENT_BEFORE_SWAP_MODEL_ID.to_owned(),
                runtime_model_policy(
                    GENERATION_EVENT_BEFORE_SWAP_MODEL_ID,
                    "/models/generation-event-before-swap-model",
                    32,
                ),
            ),
            (
                IMAGE_MODEL_ID.to_owned(),
                image_runtime_model_policy(IMAGE_MODEL_ID, "/models/image-generation-model"),
            ),
            (
                INVALID_IMAGE_MODEL_ID.to_owned(),
                image_runtime_model_policy(INVALID_IMAGE_MODEL_ID, "/models/invalid-model"),
            ),
            (
                DELAYED_POLICY_ACK_MODEL_ID.to_owned(),
                runtime_model_policy(
                    DELAYED_POLICY_ACK_MODEL_ID,
                    "/models/delayed-policy-ack-model",
                    64,
                ),
            ),
            (
                DELAYED_IMAGE_POLICY_ACK_MODEL_ID.to_owned(),
                image_runtime_model_policy(
                    DELAYED_IMAGE_POLICY_ACK_MODEL_ID,
                    "/models/delayed-image-policy-ack-model",
                ),
            ),
        ])),
        WorkerStartupConfiguration {
            configuration_generation: "test-configuration-generation".to_owned(),
            global_prompt_cache_root_directory: temporary_log_directory.path().join("prompt-cache"),
            global_prompt_cache_maximum_size_bytes: 50_000_000_000,
            persistent_prompt_cache_enabled: true,
            configured_maximum_mlx_memory_bytes: None,
            performance_attribution_enabled: false,
            logging_directory: temporary_log_directory.path().to_path_buf(),
            logging_level: WorkerLogLevel::Warn,
            retained_log_file_count: 7,
        },
    )
    .await
    .expect("the idle worker should launch");
    wait_for_ready_worker(&worker_handle).await;
    worker_handle
}

fn image_runtime_model_policy(model_id: &str, model_directory: &str) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: PathBuf::from(model_directory),
        generation_defaults: astronomical_supervisor::RuntimeModelGenerationDefaults {
            maximum_output_tokens: 0,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 0,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Flux2Klein(
            WorkerFlux2KleinModelConfiguration {
                model_id: model_id.to_owned(),
                model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                artifact_revision: "fixture-revision".to_owned(),
            },
        ),
    }
}

fn image_generation_command(model_id: &str, request_id: u64) -> ImageGenerationCommand {
    ImageGenerationCommand {
        request_id: RequestId::new(request_id),
        model: model_id.to_owned(),
        prompt: "A moonlit balcony scene from Romeo and Juliet".to_owned(),
        settings: ImageGenerationSettings {
            width_pixels: 1_024,
            height_pixels: 1_024,
            steps: 4,
            guidance_thousandths: 1_000,
            seed: 7,
        },
    }
}

fn runtime_model_policy(
    model_id: &str,
    model_directory: &str,
    maximum_output_tokens: u32,
) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: PathBuf::from(model_directory),
        generation_defaults: astronomical_supervisor::RuntimeModelGenerationDefaults {
            maximum_output_tokens: u16::try_from(maximum_output_tokens).unwrap_or(u16::MAX),
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 2_048,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
                model_id: model_id.to_owned(),
                maximum_context_tokens: 2_048,
                maximum_output_tokens,
                chunking: WorkerChunkingConfiguration {
                    fixed_prompt_processing_chunk_size_tokens: 256,
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
                    full_attention_key_value_growth_tokens: 256,
                    speculative_prefill_draft_forward_tokens: 256,
                    prefill_graph_submission_layer_interval: 0,
                    experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
                    experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                    prompt_cache_block_tokens: None,
                    prompt_cache_common_prefix_stride_blocks: 4,
                },
                mtp_enabled: true,
                mtp_draft_depth: None,
                speculative_prefill: None,
            },
        ),
    }
}

async fn wait_for_ready_worker(worker_handle: &WorkerHandle) {
    let readiness_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let worker_health_status = worker_handle.worker_health_snapshot().status;
        if worker_health_status == WorkerHealthStatus::Ready {
            return;
        }
        assert!(
            Instant::now() < readiness_deadline,
            "idle worker did not become ready; last status was {worker_health_status:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

pub(super) async fn assert_generation_completed(
    generation_events: &mut tokio::sync::mpsc::Receiver<ChatGenerationStreamEvent>,
) {
    let generation_event = timeout(Duration::from_secs(2), generation_events.recv())
        .await
        .expect("the generation should complete before the timeout")
        .expect("the generation stream should contain a completion event");
    assert!(matches!(
        generation_event,
        ChatGenerationStreamEvent::Completed {
            reason: ChatGenerationCompletionReason::EndOfSequence,
            ..
        }
    ));
}

pub(super) fn chat_command(model_id: &str, request_id: u64) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: model_id.to_owned(),
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
        qwen_thinking_channel_seed: None,
    }
}
