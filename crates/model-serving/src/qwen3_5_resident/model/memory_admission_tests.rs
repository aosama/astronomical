//! Beside-source unit tests for the pure context-admission helpers: the
//! reclamation delta measurement and the post-reclamation re-projection.

use super::{
    ContextAdmissionReservations, ContextReclamationDelta, ExpertWeightMemoryCacheStatistics,
    MemoryAdmissionDecision, ReclaimedContextAdmissionProjection,
    measure_context_reclamation_delta, project_reclaimed_context_admission,
};
use crate::MemoryBoundary;

fn cache_statistics_with_resident_payload_and_evictions(
    resident_payload_byte_count: u64,
    eviction_count: u64,
) -> ExpertWeightMemoryCacheStatistics {
    ExpertWeightMemoryCacheStatistics {
        resident_payload_byte_count,
        eviction_count,
        ..ExpertWeightMemoryCacheStatistics::default()
    }
}

/// Projection addends: 100 live + 200 context + 50 page + 25 workspace = 375 peak.
fn reservations_with_active_memory_ceiling_bytes(
    active_memory_ceiling_bytes: usize,
) -> ContextAdmissionReservations {
    ContextAdmissionReservations {
        context_reservation_bytes: 200,
        maximum_expert_page_reservation_bytes: 50,
        temporary_workspace_reservation_bytes: 25,
        active_memory_ceiling_bytes,
    }
}

#[test]
fn should_report_reclaimed_payload_and_overshoot_when_reclamation_exceeds_the_target() {
    let cache_statistics_before_reclamation =
        cache_statistics_with_resident_payload_and_evictions(2_000, 3);
    let cache_statistics_after_reclamation =
        cache_statistics_with_resident_payload_and_evictions(500, 7);
    let context_reclamation_delta = measure_context_reclamation_delta(
        &cache_statistics_before_reclamation,
        &cache_statistics_after_reclamation,
        1_000,
    );
    assert_eq!(
        context_reclamation_delta,
        ContextReclamationDelta {
            actual_reclaimed_expert_payload_bytes: 1_500,
            reclamation_overshoot_bytes: 500,
        }
    );
}

#[test]
fn should_report_no_reclaimed_payload_when_the_cache_grew_during_reclamation() {
    let cache_statistics_before_reclamation =
        cache_statistics_with_resident_payload_and_evictions(400, 3);
    let cache_statistics_after_reclamation =
        cache_statistics_with_resident_payload_and_evictions(500, 7);
    let context_reclamation_delta = measure_context_reclamation_delta(
        &cache_statistics_before_reclamation,
        &cache_statistics_after_reclamation,
        1_000,
    );
    assert_eq!(
        context_reclamation_delta,
        ContextReclamationDelta {
            actual_reclaimed_expert_payload_bytes: 0,
            reclamation_overshoot_bytes: 0,
        }
    );
}

#[test]
fn should_report_zero_overshoot_when_reclaimed_payload_exactly_matches_the_target() {
    let cache_statistics_before_reclamation =
        cache_statistics_with_resident_payload_and_evictions(1_000, 3);
    let cache_statistics_after_reclamation =
        cache_statistics_with_resident_payload_and_evictions(0, 3);
    let context_reclamation_delta = measure_context_reclamation_delta(
        &cache_statistics_before_reclamation,
        &cache_statistics_after_reclamation,
        1_000,
    );
    assert_eq!(
        context_reclamation_delta,
        ContextReclamationDelta {
            actual_reclaimed_expert_payload_bytes: 1_000,
            reclamation_overshoot_bytes: 0,
        }
    );
}

#[test]
fn should_report_every_reclaimed_byte_as_overshoot_when_the_target_is_zero() {
    let cache_statistics_before_reclamation =
        cache_statistics_with_resident_payload_and_evictions(700, 3);
    let cache_statistics_after_reclamation =
        cache_statistics_with_resident_payload_and_evictions(100, 3);
    let context_reclamation_delta = measure_context_reclamation_delta(
        &cache_statistics_before_reclamation,
        &cache_statistics_after_reclamation,
        0,
    );
    assert_eq!(
        context_reclamation_delta,
        ContextReclamationDelta {
            actual_reclaimed_expert_payload_bytes: 600,
            reclamation_overshoot_bytes: 600,
        }
    );
}

#[test]
fn should_admit_the_reclaimed_context_when_the_projection_fits_the_ceiling() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(400),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 0,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Admit,
        }
    );
}

#[test]
fn should_admit_the_reclaimed_context_when_the_projection_exactly_meets_the_ceiling() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(375),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 0,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Admit,
        }
    );
}

#[test]
fn should_reject_the_reclaimed_context_when_no_retained_payload_covers_the_shortfall() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(100),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 0,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: 275,
            },
        }
    );
}

#[test]
fn should_request_reclamation_when_retained_payload_fully_covers_the_shortfall() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(100),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 1_000,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Reclaim {
                required_bytes: 275,
            },
        }
    );
}

#[test]
fn should_request_reclamation_when_retained_payload_exactly_matches_the_shortfall() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(100),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 275,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Reclaim {
                required_bytes: 275,
            },
        }
    );
}

#[test]
fn should_reject_the_reclaimed_context_when_retained_payload_is_one_byte_short() {
    let post_reclamation_projection = project_reclaimed_context_admission(
        reservations_with_active_memory_ceiling_bytes(100),
        /* active_memory_bytes_after_reclamation */ 100,
        /* retained_expert_payload_bytes_after_reclamation */ 274,
    );
    assert_eq!(
        post_reclamation_projection,
        ReclaimedContextAdmissionProjection {
            projected_active_memory_bytes_after_reclamation: 375,
            post_reclamation_decision: MemoryAdmissionDecision::Reject {
                boundary: MemoryBoundary::StableActiveCeiling,
                shortfall_bytes: 1,
            },
        }
    );
}
