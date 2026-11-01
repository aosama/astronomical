//! The observatory expert-streaming stats endpoint (`GET /v1/experts/stats`).
//!
//! The document is assembled purely from daemon-accumulated worker events so
//! the handler stays a thin reader and every derived number (hit rate, read
//! speed, the time-saved counterfactual) lives in a pure, unit-tested
//! function. Numbers are raw measurements; the one estimate on the page —
//! storage time saved — is labelled as such and shipped with its inputs so
//! the console can show the math.

use crate::application::ApplicationState;
use crate::worker::health::expert_streaming::ExpertStreamingObservatory;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

/// Pure counterfactual: nanoseconds storage reads would have added if the
/// assignments served from retained RAM had missed instead.
///
/// The estimate divides the measured storage-read elapsed time by the storage
/// assignment count — "if RAM hits had gone to storage at this machine's
/// actual measured storage speed" — so it is calibrated per machine and uses
/// no synthetic benchmark. Zero storage misses yields zero: without a
/// measured miss there is no honest speed to extrapolate from.
#[must_use]
pub(crate) fn estimate_storage_time_saved_nanoseconds(
    retained_route_assignment_count: u64,
    storage_route_assignment_count: u64,
    expert_source_read_elapsed_nanoseconds: u64,
) -> u64 {
    if storage_route_assignment_count == 0 || retained_route_assignment_count == 0 {
        return 0;
    }
    let average_storage_assignment_nanoseconds =
        expert_source_read_elapsed_nanoseconds / storage_route_assignment_count;
    average_storage_assignment_nanoseconds.saturating_mul(retained_route_assignment_count)
}

/// Pure: share of routed assignments served from retained RAM (0.0–1.0).
#[must_use]
pub(crate) fn resident_route_hit_rate(
    retained_route_assignment_count: u64,
    storage_route_assignment_count: u64,
) -> f64 {
    let total_assignment_count = retained_route_assignment_count + storage_route_assignment_count;
    if total_assignment_count == 0 {
        return 0.0;
    }
    f64::from(u32::try_from(retained_route_assignment_count).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(total_assignment_count).unwrap_or(u32::MAX))
}

/// Pure: expert-library bytes that are not in RAM. A resident payload larger
/// than the measured library saturates at zero rather than wrapping.
#[must_use]
pub(crate) fn expert_payload_left_on_ssd_bytes(
    complete_expert_payload_byte_count: u64,
    resident_expert_payload_bytes: u64,
) -> u64 {
    complete_expert_payload_byte_count.saturating_sub(resident_expert_payload_bytes)
}

/// Pure: bytes per second the machine actually served expert reads at.
#[must_use]
pub(crate) fn measured_storage_read_bytes_per_second(
    expert_source_read_byte_count: u64,
    expert_source_read_elapsed_nanoseconds: u64,
) -> u64 {
    if expert_source_read_elapsed_nanoseconds == 0 {
        return 0;
    }
    let elapsed_seconds = expert_source_read_elapsed_nanoseconds as f64 / 1_000_000_000.0;
    (expert_source_read_byte_count as f64 / elapsed_seconds) as u64
}

/// Pure assembly of the observatory expert stats document.
#[must_use]
pub(crate) fn expert_stats_document(
    expert_streaming_observatory: &ExpertStreamingObservatory,
) -> Value {
    let Some(latest_stats) = expert_streaming_observatory.latest_stats.as_ref() else {
        return json!({
            "available": false,
            "mode_transitions": mode_transition_entries(expert_streaming_observatory),
        });
    };
    let storage_time_saved_estimate_nanoseconds = estimate_storage_time_saved_nanoseconds(
        latest_stats.retained_route_assignment_count,
        latest_stats.storage_route_assignment_count,
        latest_stats.expert_source_read_elapsed_nanoseconds,
    );
    json!({
        "available": true,
        "model_id": latest_stats.model_id,
        "expert_memory_mode": latest_stats.expert_memory_mode,
        "residency": {
            "total_layer_count": latest_stats.expert_residency.total_layer_count,
            "resident_expert_count": latest_stats.expert_residency.resident_expert_count,
            "resident_expert_payload_bytes": latest_stats.expert_residency.resident_expert_payload_bytes,
            "complete_expert_payload_byte_count":
                latest_stats.expert_residency.complete_expert_payload_byte_count,
            "expert_capacity": latest_stats.expert_residency.expert_capacity,
        },
        "library": {
            "complete_expert_payload_byte_count":
                latest_stats.expert_residency.complete_expert_payload_byte_count,
            "expert_capacity": latest_stats.expert_residency.expert_capacity,
            "expert_payload_left_on_ssd_bytes": expert_payload_left_on_ssd_bytes(
                latest_stats.expert_residency.complete_expert_payload_byte_count,
                latest_stats.expert_residency.resident_expert_payload_bytes,
            ),
        },
        "value": {
            "retained_route_assignment_count": latest_stats.retained_route_assignment_count,
            "storage_route_assignment_count": latest_stats.storage_route_assignment_count,
            "resident_route_hit_rate": resident_route_hit_rate(
                latest_stats.retained_route_assignment_count,
                latest_stats.storage_route_assignment_count,
            ),
            "storage_time_saved_estimate_nanoseconds":
                storage_time_saved_estimate_nanoseconds,
            "storage_time_saved_estimate_seconds":
                storage_time_saved_estimate_nanoseconds / 1_000_000_000,
            "avoided_complete_layer_source_bytes":
                latest_stats.avoided_complete_layer_source_bytes,
            "mandatory_prefill_source_bytes": latest_stats.mandatory_prefill_source_bytes,
            "mandatory_decode_source_bytes": latest_stats.mandatory_decode_source_bytes,
            "expert_source_read_byte_count": latest_stats.expert_source_read_byte_count,
            "expert_source_read_call_count": latest_stats.expert_source_read_call_count,
            "expert_source_read_elapsed_nanoseconds":
                latest_stats.expert_source_read_elapsed_nanoseconds,
            "measured_storage_read_bytes_per_second": measured_storage_read_bytes_per_second(
                latest_stats.expert_source_read_byte_count,
                latest_stats.expert_source_read_elapsed_nanoseconds,
            ),
            "hot_route_fully_covered_token_count":
                latest_stats.hot_route_fully_covered_token_count,
            "hot_route_partially_covered_token_count":
                latest_stats.hot_route_partially_covered_token_count,
            "hot_route_missed_token_count": latest_stats.hot_route_missed_token_count,
            "session_prompt_token_count":
                expert_streaming_observatory.accumulated_prompt_token_count,
            "session_generated_token_count":
                expert_streaming_observatory.accumulated_generated_token_count,
        },
        "layer_topology": layer_topology_entries(latest_stats),
        "mode_transitions": mode_transition_entries(expert_streaming_observatory),
        "updated_at_unix_millis": expert_streaming_observatory.updated_at_unix_millis,
    })
}

/// Pure: topology rows for the observatory memory map.
fn layer_topology_entries(
    latest_stats: &astronomical_ipc_protocol::WorkerExpertStreamingStats,
) -> Vec<Value> {
    latest_stats
        .layer_topology
        .iter()
        .map(|layer_topology| {
            json!({
                "layer_index": layer_topology.layer_index,
                "retained_expert_ids": layer_topology.retained_expert_ids,
                "retained_payload_bytes": layer_topology.retained_payload_bytes,
                "is_complete_layer": layer_topology.is_complete_layer,
                "covered_demand": layer_topology.covered_demand,
            })
        })
        .collect()
}

/// Pure: oldest-first decision-feed entries.
fn mode_transition_entries(
    expert_streaming_observatory: &ExpertStreamingObservatory,
) -> Vec<Value> {
    expert_streaming_observatory
        .mode_transitions
        .iter()
        .map(|mode_transition| {
            json!({
                "observed_at_unix_millis": mode_transition.observed_at_unix_millis,
                "previous_mode": mode_transition.previous_mode,
                "next_mode": mode_transition.next_mode,
            })
        })
        .collect()
}

pub(crate) async fn expert_stats(State(application_state): State<ApplicationState>) -> Response {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    Json(expert_stats_document(
        &worker_health_snapshot.expert_streaming_observatory,
    ))
    .into_response()
}
