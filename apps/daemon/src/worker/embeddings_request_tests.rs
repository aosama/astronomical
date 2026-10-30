//! Unit contracts for the pure embeddings admission gate: the capability
//! check that guards command dispatch.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{WorkerEmbeddingCapabilities, WorkerModelCapabilities};

use super::embeddings_command_supported;
use crate::WorkerHealthSnapshot;

fn ready_snapshot_with_embedding_width(vector_width: u32) -> WorkerHealthSnapshot {
    WorkerHealthSnapshot::ready_with_model(
        String::from("embedding-model"),
        WorkerModelCapabilities::embeddings(WorkerEmbeddingCapabilities {
            vector_width,
            max_input_tokens: 512,
        }),
    )
}

#[test]
fn should_accept_dimensions_within_the_advertised_vector_width() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot_with_embedding_width(1_024)));

    let is_supported = embeddings_command_supported(&health_snapshot, Some(512));

    assert!(
        is_supported,
        "truncating dimensions must be accepted against a wider model"
    );
}

#[test]
fn should_accept_a_request_without_explicit_dimensions() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot_with_embedding_width(1_024)));

    let is_supported = embeddings_command_supported(&health_snapshot, None);

    assert!(
        is_supported,
        "a request without dimensions uses the native vector width"
    );
}

#[test]
fn should_accept_dimensions_exactly_at_the_advertised_vector_width() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot_with_embedding_width(1_024)));

    let is_supported = embeddings_command_supported(&health_snapshot, Some(1_024));

    assert!(
        is_supported,
        "dimensions equal to the vector width are not a truncation violation"
    );
}

#[test]
fn should_reject_dimensions_beyond_the_advertised_vector_width() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot_with_embedding_width(1_024)));

    let is_supported = embeddings_command_supported(&health_snapshot, Some(2_048));

    assert!(
        !is_supported,
        "dimensions wider than the model can produce must be refused"
    );
}

#[test]
fn should_reject_a_model_without_embedding_capabilities() {
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("non-embedding-model"),
        WorkerModelCapabilities {
            chat: None,
            image_generation: None,
            embeddings: None,
        },
    )));

    let is_supported = embeddings_command_supported(&health_snapshot, None);

    assert!(
        !is_supported,
        "a model without an advertised embedding surface must refuse every command"
    );
}

#[test]
fn should_reject_a_zero_advertised_vector_width() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot_with_embedding_width(0)));

    let is_supported = embeddings_command_supported(&health_snapshot, Some(1));

    assert!(
        !is_supported,
        "a non-positive advertised vector width cannot serve any command"
    );
}
