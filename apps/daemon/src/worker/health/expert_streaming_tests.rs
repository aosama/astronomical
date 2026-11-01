//! Unit contract for daemon-side expert-streaming observatory accumulation.

use astronomical_ipc_protocol::{
    ExpertMemoryMode, WorkerEvent, WorkerExpertLayerTopology, WorkerExpertResidencySnapshot,
    WorkerExpertStreamingStats,
};

use super::expert_streaming::{
    ExpertStreamingObservatory, MAXIMUM_MODE_TRANSITION_HISTORY, expert_streaming_stats_from_event,
};
use super::publish_expert_streaming_stats;

fn stats_snapshot(
    expert_memory_mode: ExpertMemoryMode,
    prompt_tokens: u64,
) -> WorkerExpertStreamingStats {
    WorkerExpertStreamingStats {
        model_id: Some(String::from("qwen35-moe")),
        expert_memory_mode,
        expert_residency: WorkerExpertResidencySnapshot {
            total_layer_count: 60,
            resident_expert_count: 3_200,
            resident_expert_payload_bytes: 40_000_000_000,
            complete_expert_payload_byte_count: 90_000_000_000,
            expert_capacity: 256,
            layer_seating: astronomical_ipc_protocol::ExpertLayerSeatingCounts::default(),
        },
        layer_topology: vec![WorkerExpertLayerTopology {
            layer_index: 0,
            retained_expert_ids: vec![1, 2, 3],
            retained_payload_bytes: 1_048_576,
            is_complete_layer: false,
            covered_demand: 42,
        }],
        retained_route_assignment_count: 10,
        storage_route_assignment_count: 4,
        hot_route_fully_covered_token_count: 5,
        hot_route_partially_covered_token_count: 2,
        hot_route_missed_token_count: 1,
        expert_source_read_call_count: 8,
        expert_source_read_byte_count: 900_000,
        expert_source_read_elapsed_nanoseconds: 400_000_000,
        avoided_complete_layer_source_bytes: 50_000_000,
        mandatory_prefill_source_bytes: 30_000_000,
        mandatory_decode_source_bytes: 4_000_000,
        prompt_token_count: prompt_tokens,
        generated_token_count: 100,
    }
}

#[test]
fn should_accumulate_token_totals_across_stats_events() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Hybrid, 500), 1_000);
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Hybrid, 700), 2_000);
    assert_eq!(observatory.accumulated_prompt_token_count, 1_200);
    assert_eq!(observatory.accumulated_generated_token_count, 200);
    assert_eq!(observatory.updated_at_unix_millis, Some(2_000));
}

#[test]
fn should_record_a_transition_only_when_the_mode_actually_changes() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Paged, 10), 1_000);
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Hybrid, 10), 2_000);
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Hybrid, 10), 3_000);
    assert_eq!(observatory.mode_transitions.len(), 2);
    assert_eq!(observatory.mode_transitions[0].previous_mode, None);
    assert_eq!(
        observatory.mode_transitions[0].next_mode,
        ExpertMemoryMode::Paged
    );
    assert_eq!(
        observatory.mode_transitions[1].previous_mode,
        Some(ExpertMemoryMode::Paged)
    );
    assert_eq!(
        observatory.mode_transitions[1].next_mode,
        ExpertMemoryMode::Hybrid
    );
}

#[test]
fn should_reconcile_out_of_band_mode_changes_into_the_latest_snapshot() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Paged, 10), 1_000);
    observatory.record_mode_change(ExpertMemoryMode::Resident, 1_500);
    let latest_stats = observatory
        .latest_stats
        .as_ref()
        .expect("latest stats must survive a mode change");
    assert_eq!(latest_stats.expert_memory_mode, ExpertMemoryMode::Resident);
    assert_eq!(observatory.mode_transitions.len(), 2);
}

#[test]
fn should_ignore_a_mode_change_that_matches_the_latest_snapshot() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(stats_snapshot(ExpertMemoryMode::Hybrid, 10), 1_000);
    observatory.record_mode_change(ExpertMemoryMode::Hybrid, 1_200);
    assert_eq!(observatory.mode_transitions.len(), 1);
}

#[test]
fn should_bound_the_transition_ring_by_dropping_the_oldest() {
    let mut observatory = ExpertStreamingObservatory::empty();
    for transition_index in 0..(MAXIMUM_MODE_TRANSITION_HISTORY + 5) {
        let mode_for_index = if transition_index % 2 == 0 {
            ExpertMemoryMode::Paged
        } else {
            ExpertMemoryMode::Hybrid
        };
        observatory.record_stats(
            stats_snapshot(mode_for_index, 1),
            1_000 + transition_index as u64,
        );
    }
    assert_eq!(
        observatory.mode_transitions.len(),
        MAXIMUM_MODE_TRANSITION_HISTORY
    );
    // Five transitions were dropped; the oldest survivor is index 5, whose
    // timestamp carries the 1_000 base used across the loop.
    assert_eq!(
        observatory.mode_transitions[0].observed_at_unix_millis,
        1_005
    );
}

#[test]
fn should_extract_stats_only_from_the_matching_event() {
    let matching_event = WorkerEvent::ExpertStreamingStats {
        stats: stats_snapshot(ExpertMemoryMode::Resident, 1),
    };
    assert!(expert_streaming_stats_from_event(&matching_event).is_some());
    assert!(
        expert_streaming_stats_from_event(&WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Resident
        })
        .is_none()
    );
}

#[test]
fn should_carry_the_observatory_across_replacement_for_the_same_model() {
    use std::sync::{Arc, RwLock};

    use astronomical_ipc_protocol::WorkerModelCapabilities;

    use crate::worker::health::{WorkerHealthSnapshot, publish_health};

    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("ornith-35b"),
        WorkerModelCapabilities {
            chat: None,
            image_generation: None,
            embeddings: None,
        },
    )));
    publish_expert_streaming_stats(
        &health_snapshot,
        WorkerEvent::ExpertStreamingStats {
            stats: stats_snapshot(ExpertMemoryMode::Hybrid, 500),
        },
    );

    // An idle transition rebuilds the whole snapshot for the same model; the
    // accumulated observatory must survive it or every feed entry would
    // restart from `previous_mode: None`.
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_with_model(
            String::from("ornith-35b"),
            WorkerModelCapabilities {
                chat: None,
                image_generation: None,
                embeddings: None,
            },
        ),
    );

    let carried_observatory = health_snapshot
        .read()
        .expect("the health snapshot should stay readable")
        .expert_streaming_observatory
        .clone();
    assert!(carried_observatory.latest_stats.is_some());
    assert_eq!(carried_observatory.accumulated_prompt_token_count, 500);
    assert_eq!(carried_observatory.mode_transitions.len(), 1);
}

#[test]
fn should_carry_the_observatory_across_a_model_less_idle_replacement() {
    use std::sync::{Arc, RwLock};

    use astronomical_ipc_protocol::WorkerModelCapabilities;

    use crate::worker::health::{WorkerHealthSnapshot, publish_health};

    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("ornith-35b"),
        WorkerModelCapabilities {
            chat: None,
            image_generation: None,
            embeddings: None,
        },
    )));
    publish_expert_streaming_stats(
        &health_snapshot,
        WorkerEvent::ExpertStreamingStats {
            stats: stats_snapshot(ExpertMemoryMode::Hybrid, 500),
        },
    );

    // Idle transitions rebuild through the model-less constructor
    // (`ready_without_model`), so the replacement carries no model claim.
    // The live daemon showed exactly this path wiping the feed: carrying
    // here is what keeps `previous_mode` chaining between generations.
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_without_model(0),
    );

    let carried_observatory = health_snapshot
        .read()
        .expect("the health snapshot should stay readable")
        .expert_streaming_observatory
        .clone();
    assert!(carried_observatory.latest_stats.is_some());
    assert_eq!(carried_observatory.accumulated_prompt_token_count, 500);
    assert_eq!(carried_observatory.mode_transitions.len(), 1);
}

#[test]
fn should_carry_the_observatory_across_a_re_ready_without_a_published_model_claim() {
    use std::sync::{Arc, RwLock};

    use astronomical_ipc_protocol::WorkerModelCapabilities;

    use crate::worker::health::{WorkerHealthSnapshot, publish_health};

    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::ready_without_model(0)));
    publish_expert_streaming_stats(
        &health_snapshot,
        WorkerEvent::ExpertStreamingStats {
            stats: stats_snapshot(ExpertMemoryMode::Paged, 100),
        },
    );

    // The load flow re-publishes readiness with a model claim while the
    // existing snapshot still carries none. Treating that as a swap-in reset
    // `latest_stats` between phase-boundary stats events, which is why the
    // live feed showed `previous_mode: None` on every entry.
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_with_model(
            String::from("ornith-35b"),
            WorkerModelCapabilities {
                chat: None,
                image_generation: None,
                embeddings: None,
            },
        ),
    );

    let carried_observatory = health_snapshot
        .read()
        .expect("the health snapshot should stay readable")
        .expert_streaming_observatory
        .clone();
    assert!(carried_observatory.latest_stats.is_some());
    assert_eq!(carried_observatory.accumulated_prompt_token_count, 100);
}

#[test]
fn should_reset_the_observatory_when_the_replacement_changes_the_model() {
    use std::sync::{Arc, RwLock};

    use astronomical_ipc_protocol::WorkerModelCapabilities;

    use crate::worker::health::{WorkerHealthSnapshot, publish_health};

    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::ready_with_model(
        String::from("ornith-35b"),
        WorkerModelCapabilities {
            chat: None,
            image_generation: None,
            embeddings: None,
        },
    )));
    publish_expert_streaming_stats(
        &health_snapshot,
        WorkerEvent::ExpertStreamingStats {
            stats: stats_snapshot(ExpertMemoryMode::Hybrid, 500),
        },
    );

    // A model swap replaces the engine load; worker counters restart, so the
    // replacement's fresh observatory must win over the stale accumulation.
    publish_health(
        &health_snapshot,
        WorkerHealthSnapshot::ready_with_model(
            String::from("other-model"),
            WorkerModelCapabilities {
                chat: None,
                image_generation: None,
                embeddings: None,
            },
        ),
    );

    let fresh_observatory = health_snapshot
        .read()
        .expect("the health snapshot should stay readable")
        .expert_streaming_observatory
        .clone();
    assert!(fresh_observatory.latest_stats.is_none());
    assert_eq!(fresh_observatory.accumulated_prompt_token_count, 0);
}
