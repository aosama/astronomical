//! The worker's residency snapshot carries what the engine measured,
//! including a dense model's decoder layers, field for field.

use super::worker_expert_residency_snapshot;
use crate::inference_engine::contract::ExpertResidencyTelemetry;

#[test]
fn should_carry_a_dense_model_layer_residency_to_the_worker_snapshot() {
    let dense_layers = astronomical_ipc_protocol::WorkerDenseLayerResidency {
        decoder_layer_count: 32,
        kept_layer_count: 7,
        streamed_layer_payload_bytes: 4_437_000_000,
    };
    let telemetry = ExpertResidencyTelemetry {
        dense_layers: Some(dense_layers),
        ..ExpertResidencyTelemetry::default()
    };

    assert_eq!(
        worker_expert_residency_snapshot(telemetry).dense_layers,
        Some(dense_layers)
    );
}

#[test]
fn should_carry_no_dense_layers_for_a_sparse_model() {
    let telemetry = ExpertResidencyTelemetry {
        total_layer_count: 40,
        expert_capacity: 256,
        ..ExpertResidencyTelemetry::default()
    };

    let worker_snapshot = worker_expert_residency_snapshot(telemetry);

    assert_eq!(worker_snapshot.dense_layers, None);
    assert_eq!(worker_snapshot.total_layer_count, 40);
}
