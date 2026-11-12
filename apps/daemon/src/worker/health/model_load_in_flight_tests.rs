//! The load report a daemon publishes, and that it never outlives its load.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{ModelLoadInFlight, WorkerModelCapabilities};

use super::{
    WorkerHealthSnapshot, WorkerHealthStatus, publish_health, publish_model_load_in_flight,
};

fn no_capabilities() -> WorkerModelCapabilities {
    WorkerModelCapabilities {
        chat: None,
        image_generation: None,
        embeddings: None,
        decisions: None,
        speech: None,
    }
}

fn opening_report() -> ModelLoadInFlight {
    ModelLoadInFlight::for_model("next-model")
}

fn loading_snapshot_with_load_reported() -> Arc<RwLock<WorkerHealthSnapshot>> {
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )));
    publish_model_load_in_flight(&health_snapshot, Some(opening_report()));
    health_snapshot
}

fn published_load(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Option<ModelLoadInFlight> {
    health_snapshot
        .read()
        .expect("the snapshot reads")
        .model_load_in_flight
        .clone()
}

#[test]
fn should_start_with_no_load_reported() {
    let unavailable_snapshot = WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Loading);
    let idle_snapshot = WorkerHealthSnapshot::ready_without_model(1);
    let ready_snapshot =
        WorkerHealthSnapshot::ready_with_model(String::from("first-model"), no_capabilities());

    assert_eq!(unavailable_snapshot.model_load_in_flight, None);
    assert_eq!(idle_snapshot.model_load_in_flight, None);
    assert_eq!(ready_snapshot.model_load_in_flight, None);
}

#[test]
fn should_publish_the_newest_report() {
    let health_snapshot = loading_snapshot_with_load_reported();

    assert_eq!(published_load(&health_snapshot), Some(opening_report()));
}

#[test]
fn should_clear_on_request() {
    let health_snapshot = loading_snapshot_with_load_reported();

    publish_model_load_in_flight(&health_snapshot, None);

    assert_eq!(published_load(&health_snapshot), None);
}

#[test]
fn should_clear_when_the_model_becomes_ready() {
    let health_snapshot = loading_snapshot_with_load_reported();

    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_with_model(String::from("first-model"), no_capabilities()),
    );

    assert_eq!(published_load(&health_snapshot), None);
}

#[test]
fn should_clear_when_a_failed_load_leaves_the_worker_idle() {
    let health_snapshot = loading_snapshot_with_load_reported();

    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_without_model(1),
    );

    assert_eq!(published_load(&health_snapshot), None);
}

#[test]
fn should_clear_when_the_worker_is_lost() {
    let health_snapshot = loading_snapshot_with_load_reported();

    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
    );

    assert_eq!(published_load(&health_snapshot), None);
}

#[test]
fn should_clear_when_a_replacement_model_is_ready() {
    let previous_snapshot =
        WorkerHealthSnapshot::ready_with_model(String::from("first-model"), no_capabilities());
    let health_snapshot = Arc::new(RwLock::new(previous_snapshot.clone()));
    publish_model_load_in_flight(&health_snapshot, Some(opening_report()));
    let snapshot_during_swap = health_snapshot.read().expect("the snapshot reads").clone();

    let replacement_snapshot = WorkerHealthSnapshot::ready_with_replacement_model(
        String::from("second-model"),
        no_capabilities(),
        1,
        &snapshot_during_swap,
    );

    assert_eq!(replacement_snapshot.model_load_in_flight, None);
}
