//! Unit contracts for reading the model and configuration generation the
//! worker published. Containment and the answer channel need a live worker
//! process and run under the daemon integration journeys.

use std::sync::{Arc, RwLock};

use ipc_protocol::{WorkerModelCapabilities, WorkerRuntimeFeatureConfiguration};

use super::{loaded_model_id, published_configuration_generation};
use crate::{WorkerHealthSnapshot, WorkerHealthStatus};

fn health_snapshot_with_ready_model(model_id: &str) -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        model_id.to_owned(),
        WorkerModelCapabilities {
            chat: None,
            image_generation: None,
            embeddings: None,
            decisions: None,
            speech: None,
        },
    )))
}

fn health_snapshot_with_no_model() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )))
}

#[test]
fn should_report_the_published_ready_model() {
    let health_snapshot = health_snapshot_with_ready_model("shakespeare-sonnets");

    assert_eq!(
        loaded_model_id(&health_snapshot).as_deref(),
        Some("shakespeare-sonnets")
    );
}

#[test]
fn should_report_no_loaded_model_before_the_worker_is_ready() {
    assert_eq!(loaded_model_id(&health_snapshot_with_no_model()), None);
}

#[test]
fn should_report_no_loaded_model_when_the_health_lock_is_poisoned() {
    let health_snapshot = health_snapshot_with_ready_model("shakespeare-sonnets");
    let poisoning_snapshot = Arc::clone(&health_snapshot);
    let _poisoned_write = std::thread::spawn(move || {
        let _write_guard = poisoning_snapshot.write().expect("the lock should be free");
        panic!("poison the health lock");
    })
    .join();

    assert_eq!(loaded_model_id(&health_snapshot), None);
}

#[test]
fn should_report_the_configuration_generation_the_worker_acknowledged() {
    let health_snapshot = health_snapshot_with_no_model();
    health_snapshot
        .write()
        .expect("the snapshot writes")
        .worker_runtime_feature_configuration = Some(WorkerRuntimeFeatureConfiguration {
        configuration_generation: "generation-7".to_owned(),
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_maximum_size_bytes: 0,
        loaded_model: None,
    });

    assert_eq!(
        published_configuration_generation(&health_snapshot).as_deref(),
        Some("generation-7")
    );
}

#[test]
fn should_report_no_configuration_generation_before_any_acknowledgement() {
    assert_eq!(
        published_configuration_generation(&health_snapshot_with_no_model()),
        None
    );
}
