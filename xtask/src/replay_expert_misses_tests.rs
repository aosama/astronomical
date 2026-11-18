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
        prompt_demand_by_layer: vec![vec![9, 0, 0, 1], vec![0, 4, 0, 0], vec![0, 0, 7, 0]],
        reply_demand_by_layer: vec![vec![0, 0, 0, 0], vec![0, 0, 0, 20], vec![0, 0, 0, 0]],
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

    // A header, today's engine, eight shared rules, five whole-layer counts,
    // three prompt-seeded weights, and three rules for each of the seven
    // other divisions.
    assert_eq!(
        report_lines.len(),
        1 + 1 + 3 + 8 + 8 + 3 + 1 + 3 + 7 * 3,
        "{report_lines:#?}"
    );
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

#[test]
fn should_start_with_nothing_held() {
    let started_empty =
        super::capacity_divisions::starting_with_nothing(&saved_case(Vec::new()).replay_case);

    assert!(
        started_empty
            .held_expert_ids_by_layer_at_start
            .iter()
            .all(Vec::is_empty)
    );
}

#[test]
fn should_start_with_the_most_used_experts_up_to_the_capacity() {
    let started_warm =
        super::capacity_divisions::starting_with_the_most_used(&saved_case(Vec::new()).replay_case);

    let held_count: usize = started_warm
        .held_expert_ids_by_layer_at_start
        .iter()
        .map(Vec::len)
        .sum();
    // Room for eight; layer 2 routes to 2 and 3 twice, so both are held.
    assert_eq!(held_count, 8);
    assert!(started_warm.held_expert_ids_by_layer_at_start[2].contains(&2));
    assert!(started_warm.held_expert_ids_by_layer_at_start[2].contains(&3));
}

#[test]
fn should_divide_equally_between_layers_and_hold_none_whole() {
    let divided_case =
        super::capacity_divisions::divided_equally(&saved_case(Vec::new()).replay_case);

    // Room for eight over three layers: two each.
    assert_eq!(divided_case.layer_slot_limits, Some(vec![2, 2, 2]));
}

#[test]
fn should_start_from_the_prompts_most_demanded_experts() {
    let started_from_prompt =
        super::capacity_divisions::starting_from_demand(&saved_case(Vec::new()), 0);

    // Four experts have prompt demand and room is eight: all four start held.
    assert_eq!(
        started_from_prompt.held_expert_ids_by_layer_at_start,
        vec![vec![0, 3], vec![1], vec![2]]
    );
}

#[test]
fn should_add_earlier_replies_demand_when_asked() {
    let started_from_both =
        super::capacity_divisions::starting_from_demand(&saved_case(Vec::new()), 1);

    assert!(started_from_both.held_expert_ids_by_layer_at_start[1].contains(&3));
}

#[test]
fn should_pin_the_whole_layers_asked_for_and_share_the_rest_equally() {
    let divided_case =
        super::capacity_divisions::whole_layers_then_equal_tables(&saved_case(Vec::new()), 1);

    // Layer 0 is the only layer held whole; it stays whole and pinned, and
    // the other two share the four slots left equally.
    assert_eq!(divided_case.pinned_layer_indexes, vec![0]);
    assert_eq!(divided_case.layer_slot_limits, Some(vec![4, 2, 2]));
    assert!(divided_case.held_expert_ids_by_layer_at_start[1].is_empty());
}
