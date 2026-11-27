//! A live memory-limit change that arrives while an image is being made
//! (issue #2681): the image engine plans its components' residency against
//! the ceiling, so the change is refused as busy and the image finishes; a
//! change that lands after the last step is answered exactly once.

use super::*;

/// Swaps the scripted image model into an idle worker and starts image 950
/// with `request_steps`, returning once the model is loaded.
async fn start_image_with_steps(
    request_steps: Vec<Result<ImageGenerationEngineStep, ImageGenerationFailureReason>>,
) -> (
    ScriptedImageEngineHandles,
    ProtocolReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
    tokio::task::JoinHandle<Result<(), WorkerRuntimeError>>,
) {
    let image_engine = ScriptedImageEngine::new(vec![request_steps]);
    let updated_limits = Arc::clone(&image_engine.updated_limits);
    let (mut events, mut commands, worker) = start_worker_swapping_in(image_engine).await;
    let _swapped = next_event(&mut events).await;
    let _model_load_memory = next_event(&mut events).await;
    commands
        .send_command(&WorkerCommand::GenerateImage(image_command(950)))
        .await
        .expect("the image request should start");
    (
        ScriptedImageEngineHandles { updated_limits },
        events,
        commands,
        worker,
    )
}

struct ScriptedImageEngineHandles {
    updated_limits: Arc<std::sync::Mutex<Vec<u64>>>,
}

async fn send_limit_change(
    commands: &mut ProtocolWriter<tokio::io::WriteHalf<tokio::io::DuplexStream>>,
) {
    commands
        .send_command(&WorkerCommand::UpdateMlxMemoryLimit {
            effective_mlx_memory_ceiling_bytes: 4_000,
            configuration_generation: "generation-during-image".to_owned(),
        })
        .await
        .expect("the limit change should reach the worker");
}

#[tokio::test]
async fn should_refuse_a_memory_limit_change_during_an_image_and_finish_the_image() {
    let (engine_handles, mut events, mut commands, worker) = start_image_with_steps(vec![
        Ok(progress_step()),
        Ok(progress_step()),
        Ok(progress_step()),
        Ok(completed_step()),
    ])
    .await;
    assert!(matches!(
        next_event(&mut events).await,
        WorkerEvent::ImageGenerationProgress { .. }
    ));

    send_limit_change(&mut commands).await;

    let mut limit_answers = Vec::new();
    let mut did_complete = false;
    loop {
        match next_event(&mut events).await {
            answer @ (WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. }) => limit_answers.push(answer),
            WorkerEvent::ImageGenerationCompleted { .. } => did_complete = true,
            WorkerEvent::ImageGenerationFinalized { .. } => break,
            _ => {}
        }
    }
    assert!(did_complete, "the image must complete after the refusal");
    let [WorkerEvent::MlxMemoryLimitRejected { reason, .. }] = limit_answers.as_slice() else {
        panic!("the change should be refused once: {limit_answers:?}");
    };
    assert!(reason.contains("try again when it ends"), "{reason}");
    assert!(lock(&engine_handles.updated_limits).is_empty());
    close_worker(commands, worker).await;
}

#[tokio::test]
async fn should_answer_a_memory_limit_change_after_the_last_image_step_exactly_once() {
    let (_engine_handles, mut events, mut commands, worker) =
        start_image_with_steps(vec![Ok(completed_step())]).await;
    assert_completed_payload(next_event(&mut events).await, 950);

    send_limit_change(&mut commands).await;

    let mut limit_answer_count = 0;
    let mut was_finalized = false;
    while !(was_finalized && limit_answer_count > 0) {
        match timeout(Duration::from_secs(1), next_event(&mut events))
            .await
            .expect("the worker should answer the change and finalize the image")
        {
            WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. } => limit_answer_count += 1,
            WorkerEvent::ImageGenerationFinalized { .. } => was_finalized = true,
            _ => {}
        }
    }
    close_worker(commands, worker).await;
    while let Some(late_event) = events
        .next_event()
        .await
        .expect("the event stream should stay valid")
    {
        assert!(
            !matches!(
                late_event,
                WorkerEvent::MlxMemoryLimitChanged { .. }
                    | WorkerEvent::MlxMemoryLimitRejected { .. }
            ),
            "the change must be answered once, not again: {late_event:?}"
        );
    }
    assert_eq!(limit_answer_count, 1);
}
