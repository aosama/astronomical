use std::path::Path;

use astronomical_memory_policy::astro_expert_cache_model::{
    ReplayCase, RoutedReply, SavedReplayCase,
};

use super::capacity_divisions::{
    divided_by_demand, divided_like_todays_engine, shared_pinning_whole_layers,
};
use super::{replay_report, run};

const EXPERT_BYTES: u64 = 100;

/// Three layers of four experts, two experts per token: layer 0 held whole,
/// layer 1 partly, layer 2 nothing.
fn saved_case(held_at_end: Vec<Vec<u16>>) -> SavedReplayCase {
    let token_routes = vec![
        vec![vec![0, 1], vec![0, 1], vec![2, 3]],
        vec![vec![2, 3], vec![1, 2], vec![2, 3]],
        vec![vec![0, 3], vec![0, 1], vec![0, 1]],
    ];
    SavedReplayCase {
        workload_name: "small".to_owned(),
        replay_case: ReplayCase {
            routed_reply: RoutedReply::from_token_routes(4, &token_routes).expect("routes build"),
            expert_payload_bytes_by_layer: vec![EXPERT_BYTES; 3],
            capacity_bytes: 8 * EXPERT_BYTES,
            held_expert_ids_by_layer_at_start: vec![vec![0, 1, 2, 3], vec![0], Vec::new()],
            layer_slot_limits: None,
            pinned_layer_indexes: Vec::new(),
        },
        held_expert_ids_by_layer_at_end: held_at_end,
        warm_slot_count: Some(2),
    }
}

fn save_case_files(evidence_directory: &Path, saved_replay_case: &SavedReplayCase) {
    let case_directory = evidence_directory.join("run/case-0");
    std::fs::create_dir_all(&case_directory).expect("case directory");
    std::fs::write(
        case_directory.join("small.routes"),
        saved_replay_case
            .replay_case
            .routed_reply
            .to_recording_bytes(),
    )
    .expect("routes write");
    let case_document = serde_json::json!({
        "kept_expert_capacity_bytes": saved_replay_case.replay_case.capacity_bytes,
        "expert_payload_bytes_by_layer": saved_replay_case.replay_case.expert_payload_bytes_by_layer,
        "held_expert_ids_by_layer_at_reply_start":
            saved_replay_case.replay_case.held_expert_ids_by_layer_at_start,
        "held_expert_ids_by_layer_at_reply_end": saved_replay_case.held_expert_ids_by_layer_at_end,
        "warm_slot_count": 2,
    });
    std::fs::write(
        case_directory.join("small.case.json"),
        serde_json::to_vec(&case_document).expect("case serializes"),
    )
    .expect("case writes");
}

#[test]
fn should_report_every_division_and_rule_for_each_saved_case() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    save_case_files(
        evidence_directory.path(),
        &saved_case(vec![vec![0, 1, 2, 3], vec![0, 1], Vec::new()]),
    );

    let report_lines = replay_report(evidence_directory.path()).expect("the report runs");

    // A header, today's engine, seven shared rules, and three rules for each
    // of the two other divisions.
    assert_eq!(report_lines.len(), 1 + 1 + 7 + 3 + 3, "{report_lines:#?}");
    assert!(report_lines[0].contains("workload=small reply_tokens=3"));
    assert!(report_lines[1].contains("division=todays_engine"));
    assert!(report_lines.iter().any(|line| line.contains("Belady")));
    assert!(
        report_lines
            .iter()
            .all(|line| line.starts_with("[replay-expert-misses]"))
    );
}

#[test]
fn should_refuse_a_directory_with_no_recordings() {
    let evidence_directory = tempfile::tempdir().expect("scratch");

    let refusal = replay_report(evidence_directory.path()).expect_err("nothing to replay");

    assert!(refusal.contains("no *.case.json"), "{refusal}");
}

#[test]
fn should_refuse_to_run_without_exactly_one_evidence_directory() {
    assert!(run(&[]).is_err());
    assert!(run(&["one".to_owned(), "two".to_owned()]).is_err());
}

#[test]
fn should_divide_like_todays_engine_from_the_tables_held_at_the_end() {
    let divided_case = divided_like_todays_engine(&saved_case(vec![
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        Vec::new(),
    ]));

    // Whole stays whole, the partly held layer keeps the size it had at the
    // end, and a layer that held nothing at either end keeps nothing.
    assert_eq!(divided_case.layer_slot_limits, Some(vec![4, 3, 0]));
}

#[test]
fn should_fall_back_to_the_warm_slot_count_when_the_end_state_is_missing() {
    let divided_case = divided_like_todays_engine(&saved_case(Vec::new()));

    // Without an end state no layer reads as absent; the partly held layers
    // take the recorded warm slot count.
    assert_eq!(divided_case.layer_slot_limits, Some(vec![4, 2, 2]));
}

#[test]
fn should_pin_exactly_the_layers_held_whole_at_the_start() {
    let pinned_case = shared_pinning_whole_layers(&saved_case(Vec::new()).replay_case);

    assert_eq!(pinned_case.pinned_layer_indexes, vec![0]);
    assert_eq!(pinned_case.layer_slot_limits, None);
}

#[test]
fn should_divide_by_demand_in_proportion_to_each_layers_distinct_experts() {
    let divided_case = divided_by_demand(&saved_case(Vec::new()).replay_case);

    // Distinct experts used: layer 0 all four, layer 1 three, layer 2 four;
    // eight slots shared in proportion, rounded down.
    assert_eq!(divided_case.layer_slot_limits, Some(vec![2, 2, 2]));
}
