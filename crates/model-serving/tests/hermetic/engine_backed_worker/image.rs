//! User-journey coverage for typed image runtime ownership and reuse.

use std::sync::Arc;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ImageGenerationFailureReason, RequestId, WorkerCommand, WorkerEvent,
    WorkerLoadedModelRuntimeConfiguration, WorkerRuntimeFeatureConfiguration,
};
use astronomical_model_serving::{
    EngineBackedWorker, FLUX2_KLEIN_OFFICIAL_MODEL_ID, WorkerRuntimeError,
};
use tokio::time::timeout;

use super::chat::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::chat::support::{chat_command, worker_model_configuration};
use super::image_support::{
    ScriptedImageEngine, ScriptedRuntime, ScriptedRuntimeFactory, assert_completed_payload,
    assert_finalized_with_cleanup, close_worker, completed_step, image_command,
    image_configuration, lock, next_event, progress_step, start_idle_worker, start_worker,
    swap_image_command,
};

#[tokio::test]
async fn should_complete_an_image_before_finalization_without_duplicate_delivery_and_reuse_after_failures()
 {
    let image_engine = ScriptedImageEngine::new(vec![
        vec![Ok(progress_step()), Ok(completed_step())],
        vec![Err(ImageGenerationFailureReason::EncodingFailed {
            reason: "the scripted encoder rejected its input".to_owned(),
        })],
        vec![Ok(completed_step())],
    ]);
    let (mut events, mut commands, worker) =
        start_idle_worker(vec![Ok(ScriptedRuntime::Image(image_engine))], 8_000).await;
    assert!(matches!(
        next_event(&mut events).await,
        WorkerEvent::Idle { .. }
    ));
    commands
        .send_command(&swap_image_command())
        .await
        .expect("the image model should be selected");
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::ModelSwapped { model_id, .. } if model_id == FLUX2_KLEIN_OFFICIAL_MODEL_ID)
    );
    assert!(matches!(
        next_event(&mut events).await,
        WorkerEvent::MlxMemorySample { .. }
    ));

    let mut invalid_command = image_command(900);
    invalid_command.prompt.clear();
    commands
        .send_command(&WorkerCommand::GenerateImage(invalid_command))
        .await
        .expect("the invalid request should reach the worker");
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::ImageGenerationFailed { request_id, reason: ImageGenerationFailureReason::InvalidRequest { .. } } if request_id == RequestId::new(900))
    );
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::ImageGenerationFinalized { request_id, .. } if request_id == RequestId::new(900))
    );

    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(901)))
        .await
        .expect("the valid request should start");
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::ImageGenerationProgress { request_id, .. } if request_id == RequestId::new(901))
    );
    assert_completed_payload(next_event(&mut events).await, 901);
    assert_finalized_with_cleanup(next_event(&mut events).await, 901);

    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(902)))
        .await
        .expect("the request-scoped failure should start");
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::ImageGenerationFailed { request_id, reason: ImageGenerationFailureReason::EncodingFailed { .. } } if request_id == RequestId::new(902))
    );
    assert_finalized_with_cleanup(next_event(&mut events).await, 902);

    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(903)))
        .await
        .expect("the image runtime should remain reusable");
    assert_completed_payload(next_event(&mut events).await, 903);
    assert_finalized_with_cleanup(next_event(&mut events).await, 903);
    close_worker(commands, worker).await;
}

#[tokio::test]
async fn should_cancel_an_image_and_reuse_the_runtime() {
    let image_engine = ScriptedImageEngine::new(vec![
        vec![Ok(progress_step()), Ok(completed_step())],
        vec![Ok(completed_step())],
    ]);
    let cancellation_count = Arc::clone(&image_engine.cancellation_count);
    let (mut events, mut commands, worker) =
        start_idle_worker(vec![Ok(ScriptedRuntime::Image(image_engine))], 8_000).await;
    let _idle = next_event(&mut events).await;
    commands
        .send_command(&swap_image_command())
        .await
        .expect("swap");
    let _swapped = next_event(&mut events).await;
    let _model_load_memory = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(910)))
        .await
        .expect("generation");
    let _progress = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(910),
        })
        .await
        .expect("cancellation");
    assert_eq!(
        next_event(&mut events).await,
        WorkerEvent::ImageGenerationFailed {
            request_id: RequestId::new(910),
            reason: ImageGenerationFailureReason::Cancelled,
        }
    );
    assert_finalized_with_cleanup(next_event(&mut events).await, 910);
    assert_eq!(*lock(&cancellation_count), 1);
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(911)))
        .await
        .expect("reuse");
    assert_completed_payload(next_event(&mut events).await, 911);
    assert_finalized_with_cleanup(next_event(&mut events).await, 911);
    close_worker(commands, worker).await;
}

#[tokio::test]
async fn should_exit_fatally_without_finalizing_when_image_cancellation_cleanup_fails() {
    let image_engine = ScriptedImageEngine::failing_cancellation(
        vec![vec![Ok(progress_step())]],
        "scripted image request cleanup failed".to_owned(),
    );
    let (mut events, mut commands, worker) =
        start_idle_worker(vec![Ok(ScriptedRuntime::Image(image_engine))], 8_000).await;
    let _idle = next_event(&mut events).await;
    commands
        .send_command(&swap_image_command())
        .await
        .expect("swap");
    let _swapped = next_event(&mut events).await;
    let _model_load_memory = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(912)))
        .await
        .expect("generation");
    let _progress = next_event(&mut events).await;

    commands
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(912),
        })
        .await
        .expect("cancellation");

    let worker_error = timeout(Duration::from_secs(1), worker)
        .await
        .expect("the worker should stop after cleanup fails")
        .expect("the worker task should join")
        .expect_err("cleanup failure must be fatal");
    assert!(matches!(
        worker_error,
        WorkerRuntimeError::InferenceEngineGenerationFailed { reason }
            if reason.contains("scripted image request cleanup failed")
    ));
    assert_eq!(
        events
            .next_event()
            .await
            .expect("the event stream should remain valid"),
        None,
        "finalization must not acknowledge cleanup that failed"
    );
}

#[tokio::test]
async fn should_fail_closed_across_modalities_and_apply_an_idle_image_memory_update() {
    let image_engine = ScriptedImageEngine::new(vec![vec![Ok(completed_step())]]);
    let updated_limits = Arc::clone(&image_engine.updated_limits);
    let (mut events, mut commands, worker) = start_idle_worker(
        vec![
            Ok(ScriptedRuntime::autoregressive(
                ScriptedChatProcessor::new(),
                ScriptedChatEngine::new(),
            )),
            Ok(ScriptedRuntime::Image(image_engine)),
        ],
        8_000,
    )
    .await;
    let _idle = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/chat".to_owned(),
            model_configuration: worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("chat swap");
    let _chat_swapped = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(920)))
        .await
        .expect("unsupported image request");
    assert!(matches!(
        next_event(&mut events).await,
        WorkerEvent::ImageGenerationFailed {
            reason: ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration,
            ..
        }
    ));
    let _unsupported_finalized = next_event(&mut events).await;

    commands
        .send_command(&swap_image_command())
        .await
        .expect("image swap");
    let _image_swapped = next_event(&mut events).await;
    let _image_model_load_memory = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::Generate(chat_command(921, 17)))
        .await
        .expect("unsupported chat request");
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::Failed { request_id, .. } if request_id == RequestId::new(921))
    );
    commands
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 7_000,
        })
        .await
        .expect("memory update");
    assert!(matches!(
        next_event(&mut events).await,
        WorkerEvent::MlxMemoryLimitChanged {
            effective_mlx_memory_ceiling_bytes: 7_000,
            mlx_memory_snapshot: Some(_),
            ..
        }
    ));
    assert_eq!(*lock(&updated_limits), vec![7_000]);
    close_worker(commands, worker).await;
}

#[tokio::test]
async fn should_swap_chat_image_chat_acknowledge_exact_tags_and_preserve_prior_runtime_after_failure()
 {
    let factory = ScriptedRuntimeFactory::new(vec![
        Ok(ScriptedRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new(),
        )),
        Ok(ScriptedRuntime::Image(ScriptedImageEngine::new(vec![
            vec![Ok(completed_step())],
        ]))),
        Err("the replacement artifact is invalid".to_owned()),
        Ok(ScriptedRuntime::autoregressive(
            ScriptedChatProcessor::new(),
            ScriptedChatEngine::new(),
        )),
    ]);
    let worker = EngineBackedWorker::idle_with_model_factory(factory, 8_000)
        .with_worker_runtime_feature_configuration(WorkerRuntimeFeatureConfiguration {
            configuration_generation: "generation-192".to_owned(),
            persistent_prompt_cache_enabled: false,
            prompt_cache_maximum_size_bytes: 0,
            loaded_model: None,
        });
    let (mut events, mut commands, worker_task) = start_worker(worker).await;
    let _idle = next_event(&mut events).await;
    let _idle_configuration = next_event(&mut events).await;

    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/chat".to_owned(),
            model_configuration: worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("chat swap");
    let _chat_swapped = next_event(&mut events).await;
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::RuntimeFeatureConfigurationApplied { worker_runtime_feature_configuration } if matches!(worker_runtime_feature_configuration.loaded_model, Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(_))))
    );

    commands
        .send_command(&swap_image_command())
        .await
        .expect("image swap");
    let _image_swapped = next_event(&mut events).await;
    assert!(
        matches!(next_event(&mut events).await, WorkerEvent::RuntimeFeatureConfigurationApplied { worker_runtime_feature_configuration } if worker_runtime_feature_configuration.loaded_model == Some(image_configuration().runtime_configuration()))
    );
    let _image_model_load_memory = next_event(&mut events).await;

    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/invalid".to_owned(),
            model_configuration: worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("failed swap");
    assert_eq!(
        next_event(&mut events).await,
        WorkerEvent::ModelSwapFailed {
            loaded_model_remains_ready: true,
            model_load_failure_reason: "the replacement artifact is invalid".to_owned(),
        }
    );
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(930)))
        .await
        .expect("the prior image runtime should survive");
    assert_completed_payload(next_event(&mut events).await, 930);
    assert_finalized_with_cleanup(next_event(&mut events).await, 930);

    commands
        .send_command(&WorkerCommand::SwapModel {
            model_directory: "/models/chat-again".to_owned(),
            model_configuration: worker_model_configuration("example/scripted-chat"),
        })
        .await
        .expect("chat replacement");
    let _chat_swapped_again = next_event(&mut events).await;
    let _chat_configuration = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::Generate(chat_command(931, 19)))
        .await
        .expect("chat after image");
    while !matches!(next_event(&mut events).await, WorkerEvent::Completed { request_id, .. } if request_id == RequestId::new(931))
    {
    }
    close_worker(commands, worker_task).await;
}

#[tokio::test]
async fn should_report_a_bounded_actionable_image_engine_load_failure() {
    let detailed_failure = format!(
        "validated transformer inventory mismatch: {}",
        "x".repeat(600)
    );
    let image_engine = ScriptedImageEngine::failing_load(detailed_failure);
    let (mut events, mut commands, worker) =
        start_idle_worker(vec![Ok(ScriptedRuntime::Image(image_engine))], 8_000).await;
    let _idle = next_event(&mut events).await;

    commands
        .send_command(&swap_image_command())
        .await
        .expect("the rejected image swap should reach the worker");
    let WorkerEvent::ModelSwapFailed {
        loaded_model_remains_ready,
        model_load_failure_reason,
    } = next_event(&mut events).await
    else {
        panic!("the image load rejection should remain a model-swap failure");
    };

    assert!(!loaded_model_remains_ready);
    assert!(model_load_failure_reason.starts_with(
        "image engine initialization failed: validated transformer inventory mismatch"
    ));
    assert!(model_load_failure_reason.chars().count() <= 512);
    close_worker(commands, worker).await;
}
