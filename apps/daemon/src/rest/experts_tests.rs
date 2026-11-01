//! Unit contract for the observatory expert stats document assembly.

use astronomical_ipc_protocol::{
    ExpertMemoryMode, WorkerExpertLayerTopology, WorkerExpertResidencySnapshot,
    WorkerExpertStreamingStats,
};

use super::experts::{
    estimate_storage_time_saved_nanoseconds, expert_payload_left_on_ssd_bytes,
    expert_stats_document, measured_storage_read_bytes_per_second, resident_route_hit_rate,
};
use crate::worker::health::expert_streaming::ExpertStreamingObservatory;

fn populated_stats_snapshot() -> WorkerExpertStreamingStats {
    WorkerExpertStreamingStats {
        model_id: Some(String::from("qwen35-122b-moe")),
        expert_memory_mode: ExpertMemoryMode::Hybrid,
        expert_residency: WorkerExpertResidencySnapshot {
            total_layer_count: 60,
            resident_expert_count: 3_200,
            resident_expert_payload_bytes: 38_000_000_000,
            complete_expert_payload_byte_count: 120_000_000_000,
            expert_capacity: 256,
            layer_seating: astronomical_ipc_protocol::ExpertLayerSeatingCounts::default(),
        },
        layer_topology: vec![
            WorkerExpertLayerTopology {
                layer_index: 4,
                retained_expert_ids: vec![7, 19, 200],
                retained_payload_bytes: 12_000_000,
                is_complete_layer: false,
                covered_demand: 410,
            },
            WorkerExpertLayerTopology {
                layer_index: 9,
                retained_expert_ids: (0..256).collect(),
                retained_payload_bytes: 1_000_000_000,
                is_complete_layer: true,
                covered_demand: 9_000,
            },
        ],
        retained_route_assignment_count: 3_000,
        storage_route_assignment_count: 1_000,
        hot_route_fully_covered_layer_token_count: 700,
        hot_route_partially_covered_layer_token_count: 200,
        hot_route_missed_layer_token_count: 100,
        expert_source_read_call_count: 2_400,
        expert_source_read_byte_count: 96_000_000_000,
        expert_source_read_elapsed_nanoseconds: 8_000_000_000,
        prefill_source_read_elapsed_nanoseconds: 7_000_000_000,
        decode_source_read_elapsed_nanoseconds: 1_000_000_000,
        expert_source_physical_disk_read_bytes: Some(64_000_000_000),
        avoided_complete_layer_source_bytes: 120_000_000_000,
        mandatory_prefill_source_bytes: 80_000_000_000,
        mandatory_decode_source_bytes: 9_000_000_000,
        prompt_token_count: 5_000,
        generated_token_count: 600,
    }
}

#[test]
fn should_report_unavailable_before_the_first_worker_event() {
    let observatory = ExpertStreamingObservatory::empty();
    let document = expert_stats_document(&observatory);
    assert_eq!(document["available"], serde_json::json!(false));
}

#[test]
fn should_assemble_the_full_document_from_one_snapshot() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(populated_stats_snapshot(), 1_735_000_000_000);
    let document = expert_stats_document(&observatory);
    assert_eq!(document["available"], serde_json::json!(true));
    assert_eq!(document["expert_memory_mode"], serde_json::json!("hybrid"));
    assert_eq!(
        document["residency"]["resident_expert_count"],
        serde_json::json!(3_200)
    );
    assert_eq!(
        document["value"]["retained_route_assignment_count"],
        serde_json::json!(3_000)
    );
    // 3 000 retained of 4 000 total assignments is a 0.75 hit rate.
    assert_eq!(
        document["value"]["resident_route_hit_rate"],
        serde_json::json!(0.75)
    );
    // Decode-only estimate: 1 s of decode reads across 1 000 decode storage
    // assignments is 1 ms per assignment; 3 000 retained decode hits then
    // saved 3 s — the 7 s of prefill read time must not enter the estimate.
    assert_eq!(
        document["value"]["storage_time_saved_estimate_seconds"],
        serde_json::json!(3)
    );
    assert_eq!(
        document["value"]["expert_source_physical_disk_read_bytes"],
        serde_json::json!(64_000_000_000u64)
    );
    assert_eq!(
        document["value"]["hot_route_fully_covered_layer_token_count"],
        serde_json::json!(700)
    );
    // 96 GB in 8 s is 12 GB/s of measured read throughput.
    assert_eq!(
        document["value"]["measured_storage_read_bytes_per_second"],
        serde_json::json!(12_000_000_000u64)
    );
    assert_eq!(
        document["value"]["session_prompt_token_count"],
        serde_json::json!(5_000)
    );
    assert_eq!(document["layer_topology"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        document["layer_topology"][1]["is_complete_layer"],
        serde_json::json!(true)
    );
    assert_eq!(
        document["updated_at_unix_millis"],
        serde_json::json!(1_735_000_000_000u64)
    );
    assert_eq!(
        document["library"]["complete_expert_payload_byte_count"],
        serde_json::json!(120_000_000_000u64)
    );
    assert_eq!(
        document["library"]["expert_capacity"],
        serde_json::json!(256)
    );
    assert_eq!(
        document["library"]["expert_payload_left_on_ssd_bytes"],
        serde_json::json!(82_000_000_000u64)
    );
}

#[test]
fn should_include_the_mode_transition_feed_oldest_first() {
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(populated_stats_snapshot(), 1_000);
    let mut promoted_snapshot = populated_stats_snapshot();
    promoted_snapshot.expert_memory_mode = ExpertMemoryMode::Resident;
    observatory.record_stats(promoted_snapshot, 2_000);
    let document = expert_stats_document(&observatory);
    let mode_transitions = document["mode_transitions"]
        .as_array()
        .expect("feed exists");
    assert_eq!(mode_transitions.len(), 2);
    assert_eq!(
        mode_transitions[0]["next_mode"],
        serde_json::json!("hybrid")
    );
    assert_eq!(
        mode_transitions[1]["next_mode"],
        serde_json::json!("resident")
    );
}

#[test]
fn should_exclude_prefill_read_time_from_the_decode_time_saved_estimate() {
    // Decode-only on both sides: 2 s of decode reads ÷ 500 decode storage
    // assignments × 2 000 decode retained assignments = 8 s saved. The 90 s
    // of prefill read time never enters the division (issue #1715).
    assert_eq!(
        estimate_storage_time_saved_nanoseconds(2_000, 500, 2_000_000_000),
        Some(8_000_000_000)
    );
    // Zero decode misses means the estimate is unknown, not zero: without a
    // measured decode miss there is no honest speed to extrapolate from.
    assert_eq!(
        estimate_storage_time_saved_nanoseconds(1_000, 0, 5_000_000_000),
        None,
        "no measured decode miss must publish unknown, not zero"
    );
    // Edge: measured decode misses but nothing retained saves nothing known
    // to compare against — also unknown.
    assert_eq!(
        estimate_storage_time_saved_nanoseconds(0, 500, 5_000_000_000),
        None
    );
    // Boundary: one miss and one hit still extrapolates honestly.
    assert_eq!(
        estimate_storage_time_saved_nanoseconds(1, 1, 4_000_000_000),
        Some(4_000_000_000)
    );
}

#[test]
fn should_report_an_unknown_estimate_as_null_in_the_document() {
    let mut quiet_snapshot = populated_stats_snapshot();
    quiet_snapshot.storage_route_assignment_count = 0;
    quiet_snapshot.decode_source_read_elapsed_nanoseconds = 0;
    let mut observatory = ExpertStreamingObservatory::empty();
    observatory.record_stats(quiet_snapshot, 1_000);
    let document = expert_stats_document(&observatory);
    assert_eq!(
        document["value"]["storage_time_saved_estimate_nanoseconds"],
        serde_json::json!(null),
        "an unmeasured estimate must serialize as null for the console"
    );
}

#[test]
fn should_saturate_the_saved_time_estimate_for_extreme_hit_counts() {
    assert_eq!(
        estimate_storage_time_saved_nanoseconds(u64::MAX, 1, 1),
        Some(u64::MAX)
    );
}

#[test]
fn should_include_resident_layers_in_the_decode_hit_rate() {
    // All decode assignments served from retained RAM — every layer
    // resident — is a perfect 1.0 hit rate.
    assert_eq!(resident_route_hit_rate(4_000, 0), 1.0);
    // None served from RAM is a zero hit rate.
    assert_eq!(resident_route_hit_rate(0, 4_000), 0.0);
    // Empty session carries no claim either way.
    assert_eq!(resident_route_hit_rate(0, 0), 0.0);
    assert_eq!(resident_route_hit_rate(3_000, 1_000), 0.75);
}

#[test]
fn should_return_a_zero_read_speed_without_elapsed_time() {
    assert_eq!(measured_storage_read_bytes_per_second(1_000, 0), 0);
}

#[test]
fn should_report_the_library_bytes_left_on_ssd() {
    assert_eq!(expert_payload_left_on_ssd_bytes(10, 4), 6);
}

#[test]
fn should_report_zero_left_on_ssd_when_the_library_is_resident() {
    assert_eq!(expert_payload_left_on_ssd_bytes(10, 10), 0);
}

#[test]
fn should_saturate_left_on_ssd_when_resident_bytes_exceed_the_library() {
    assert_eq!(expert_payload_left_on_ssd_bytes(10, 11), 0);
    assert_eq!(expert_payload_left_on_ssd_bytes(0, 5), 0);
}
