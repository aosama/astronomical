//! Proves that live memory policy remains coherent through the next lazy model swap.

use astronomical_supervisor::{
    ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome, WorkerHealthStatus,
};

use super::worker_model_swap::{
    TELEMETRY_BEFORE_SWAP_MODEL_ID, assert_generation_completed, chat_command,
    launch_idle_worker_fixture,
};

#[tokio::test]
async fn should_load_a_model_after_a_live_memory_update_changes_the_configuration_generation() {
    let worker_handle = launch_idle_worker_fixture().await;
    let updated_configuration_generation = "generation-after-live-memory-update";
    worker_handle
        .stage_memory_configuration_generation(updated_configuration_generation.to_owned());
    let memory_update_outcome = worker_handle
        .update_mlx_memory_limit(32_000_000_000, updated_configuration_generation.to_owned())
        .await
        .expect("the live memory policy should be accepted");
    worker_handle.record_memory_configuration_generation(
        updated_configuration_generation.to_owned(),
        memory_update_outcome,
    );
    assert_eq!(memory_update_outcome, MlxMemoryLimitUpdateOutcome::Applied);

    let mut generation_events = worker_handle
        .start_chat_generation(chat_command(TELEMETRY_BEFORE_SWAP_MODEL_ID, 20))
        .await
        .expect("the model should load after the live configuration update");

    assert_generation_completed(&mut generation_events).await;
    assert_eq!(
        worker_handle.worker_health_snapshot().status,
        WorkerHealthStatus::Ready
    );
    worker_handle
        .shutdown()
        .await
        .expect("the synchronized worker should shut down cleanly");
}
