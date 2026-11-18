use super::{ReplayCaseFileError, RoutedReply, find_saved_case_documents, read_saved_replay_case};

fn write_case(directory: &std::path::Path, workload_name: &str, case_document: &str) {
    std::fs::create_dir_all(directory).expect("case directory");
    let routed_reply =
        RoutedReply::from_token_routes(8, &[vec![vec![1, 2]], vec![vec![3, 4]]]).expect("routes");
    std::fs::write(
        directory.join(format!("{workload_name}.routes")),
        routed_reply.to_recording_bytes(),
    )
    .expect("routes write");
    std::fs::write(
        directory.join(format!("{workload_name}.case.json")),
        case_document,
    )
    .expect("case writes");
}

const CASE_DOCUMENT: &str = r#"{
    "workload": "chat",
    "kept_expert_capacity_bytes": 1000,
    "expert_payload_bytes_by_layer": [100],
    "held_expert_ids_by_layer_at_reply_start": [[1]],
    "held_expert_ids_by_layer_at_reply_end": [[1, 3]],
    "warm_slot_count": 2
}"#;

#[test]
fn should_read_a_saved_case_with_its_routes_and_end_state() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(evidence_directory.path(), "chat", CASE_DOCUMENT);

    let saved_case =
        read_saved_replay_case(&evidence_directory.path().join("chat.case.json")).expect("reads");

    assert_eq!(saved_case.workload_name, "chat");
    assert_eq!(saved_case.replay_case.routed_reply.token_count(), 2);
    assert_eq!(saved_case.replay_case.capacity_bytes, 1000);
    assert_eq!(
        saved_case.replay_case.held_expert_ids_by_layer_at_start,
        vec![vec![1]]
    );
    assert_eq!(saved_case.held_expert_ids_by_layer_at_end, vec![vec![1, 3]]);
    assert_eq!(saved_case.warm_slot_count, Some(2));
    assert_eq!(saved_case.replay_case.layer_slot_limits, None);
}

#[test]
fn should_read_a_case_recorded_before_the_end_state_was_kept() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(
        evidence_directory.path(),
        "older",
        r#"{"kept_expert_capacity_bytes": 10, "expert_payload_bytes_by_layer": [1],
            "held_expert_ids_by_layer_at_reply_start": [[]]}"#,
    );

    let saved_case =
        read_saved_replay_case(&evidence_directory.path().join("older.case.json")).expect("reads");

    assert!(saved_case.held_expert_ids_by_layer_at_end.is_empty());
    assert_eq!(saved_case.warm_slot_count, None);
}

#[test]
fn should_refuse_a_case_document_that_is_not_json() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(evidence_directory.path(), "broken", "not json");

    assert!(matches!(
        read_saved_replay_case(&evidence_directory.path().join("broken.case.json")),
        Err(ReplayCaseFileError::NotACaseDocument { .. })
    ));
}

#[test]
fn should_refuse_routes_that_are_not_a_recording() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(evidence_directory.path(), "chat", CASE_DOCUMENT);
    std::fs::write(evidence_directory.path().join("chat.routes"), b"garbage").expect("write");

    assert!(matches!(
        read_saved_replay_case(&evidence_directory.path().join("chat.case.json")),
        Err(ReplayCaseFileError::NotARecording { .. })
    ));
}

#[test]
fn should_refuse_a_case_whose_routes_are_missing() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(evidence_directory.path(), "chat", CASE_DOCUMENT);
    std::fs::remove_file(evidence_directory.path().join("chat.routes")).expect("remove");

    assert!(matches!(
        read_saved_replay_case(&evidence_directory.path().join("chat.case.json")),
        Err(ReplayCaseFileError::Unreadable { .. })
    ));
}

#[test]
fn should_find_case_documents_at_any_depth_in_order_and_nothing_else() {
    let evidence_directory = tempfile::tempdir().expect("scratch");
    write_case(
        &evidence_directory.path().join("b/deep"),
        "second",
        CASE_DOCUMENT,
    );
    write_case(&evidence_directory.path().join("a"), "first", CASE_DOCUMENT);
    std::fs::write(evidence_directory.path().join("notes.txt"), b"unrelated").expect("write");

    let case_document_paths = find_saved_case_documents(evidence_directory.path()).expect("lists");

    assert_eq!(case_document_paths.len(), 2);
    assert!(case_document_paths[0].ends_with("a/first.case.json"));
    assert!(case_document_paths[1].ends_with("b/deep/second.case.json"));
}

#[test]
fn should_refuse_to_list_a_directory_that_does_not_exist() {
    let evidence_directory = tempfile::tempdir().expect("scratch");

    assert!(matches!(
        find_saved_case_documents(&evidence_directory.path().join("missing")),
        Err(ReplayCaseFileError::Unreadable { .. })
    ));
}
