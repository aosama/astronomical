//! The count a waiting client watches to tell a working worker from a stuck one.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::ImageGenerationPhase;

use super::{
    ActiveRequestProgress, WorkerHealthSnapshot, WorkerHealthStatus, clear_active_request_progress,
    publish_active_request_progress, publish_health,
};

fn fresh_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )))
}

fn published_count(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> u64 {
    health_snapshot
        .read()
        .expect("the snapshot reads")
        .request_progress_count
}

fn generation_progress(generated_token_count: u32) -> ActiveRequestProgress {
    ActiveRequestProgress::Generation {
        generated_token_count,
        maximum_output_tokens: 100,
        elapsed_millis: 1,
    }
}

#[test]
fn should_start_at_zero_before_any_progress() {
    assert_eq!(published_count(&fresh_snapshot()), 0);
}

#[test]
fn should_advance_on_every_progress_report() {
    let health_snapshot = fresh_snapshot();

    publish_active_request_progress(&health_snapshot, generation_progress(1));
    publish_active_request_progress(&health_snapshot, generation_progress(2));

    assert_eq!(published_count(&health_snapshot), 2);
}

#[test]
fn should_advance_even_when_a_report_repeats_the_same_progress() {
    let health_snapshot = fresh_snapshot();

    publish_active_request_progress(&health_snapshot, generation_progress(1));
    publish_active_request_progress(&health_snapshot, generation_progress(1));

    assert_eq!(published_count(&health_snapshot), 2);
}

#[test]
fn should_advance_for_image_generation_progress() {
    let health_snapshot = fresh_snapshot();

    publish_active_request_progress(
        &health_snapshot,
        ActiveRequestProgress::ImageGeneration {
            phase: ImageGenerationPhase::Denoising,
            completed_steps: 1,
            total_steps: 4,
            elapsed_millis: 1,
        },
    );

    assert_eq!(published_count(&health_snapshot), 1);
}

#[test]
fn should_not_change_when_a_request_finishes() {
    let health_snapshot = fresh_snapshot();
    publish_active_request_progress(&health_snapshot, generation_progress(1));

    clear_active_request_progress(&health_snapshot);

    assert_eq!(published_count(&health_snapshot), 1);
}

#[test]
fn should_forget_the_last_chunk_placement_when_a_request_ends_however_it_ended() {
    let health_snapshot = fresh_snapshot();
    crate::worker::health::record_neural_engine_chunk(
        &health_snapshot,
        astronomical_ipc_protocol::NeuralEngineChunkReport {
            placement: astronomical_ipc_protocol::NeuralEngineChunkPlacement::GpuMemoryTight,
            chunk_token_count: 2_048,
            neural_engine_token_count: 0,
            elapsed_millis: 350,
            is_prompt_warm: true,
        },
    );

    clear_active_request_progress(&health_snapshot);

    let totals = health_snapshot
        .read()
        .expect("the snapshot reads")
        .serving_session
        .neural_engine
        .clone();
    assert_eq!(totals.last_chunk_placement, None);
    assert_eq!(totals.prompt_token_count, 2_048);
}

#[test]
fn should_restart_the_count_on_a_health_replacement() {
    let health_snapshot = fresh_snapshot();
    publish_active_request_progress(&health_snapshot, generation_progress(1));

    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading),
    );

    assert_eq!(published_count(&health_snapshot), 0);
}

#[test]
fn should_wrap_rather_than_stop_at_its_limit() {
    let health_snapshot = fresh_snapshot();
    health_snapshot
        .write()
        .expect("the snapshot writes")
        .request_progress_count = u64::MAX;

    publish_active_request_progress(&health_snapshot, generation_progress(1));

    assert_eq!(published_count(&health_snapshot), 0);
}
