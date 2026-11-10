//! Beside-source tests for the catalog entry flags: the Get started pick (one per
//! role, never experimental, retiring or a helper), and the experimental and
//! retiring flags with their defaults.

use serde_json::json;

use super::download_catalog_parse_tests::{
    baseline_entry_document, empty_get_started_role_names, empty_normalized_huggingface_ids,
    entry_document_with,
};
use super::{
    DownloadCatalogEntryDocument, DownloadCatalogError, DownloadCatalogRole, parse_catalog_entry,
    validate_get_started_pick,
};

#[test]
fn should_parse_role_memory_and_flags_into_the_public_entry_type() {
    let flagged_entry = entry_document_with(&[
        ("role", json!("coding")),
        ("approximate_memory_bytes", json!(25_769_803_680_u64)),
        ("get_started", json!(true)),
        ("experimental", json!(false)),
        ("minimum_memory_bytes", json!(4_451_287_808_u64)),
    ]);

    let catalog_entry = parse_catalog_entry(
        flagged_entry,
        0,
        &mut empty_normalized_huggingface_ids(),
        &mut empty_get_started_role_names(),
    )
    .expect("a flagged coding entry should parse");

    assert_eq!(catalog_entry.role(), DownloadCatalogRole::Coding);
    assert_eq!(catalog_entry.role().as_str(), "coding");
    assert_eq!(catalog_entry.approximate_memory_bytes(), 25_769_803_680);
    assert!(catalog_entry.get_started());
    assert!(!catalog_entry.experimental());
    assert_eq!(catalog_entry.minimum_memory_bytes(), Some(4_451_287_808));
}

#[test]
fn should_default_flags_to_false_and_the_streaming_floor_to_none() {
    let plain_entry = parse_catalog_entry(
        entry_document_with(&[]),
        0,
        &mut empty_normalized_huggingface_ids(),
        &mut empty_get_started_role_names(),
    )
    .expect("an entry without optional flags should parse");

    assert!(!plain_entry.get_started());
    assert!(!plain_entry.experimental());
    assert!(!plain_entry.retiring());
    assert_eq!(plain_entry.minimum_memory_bytes(), None);
}

#[test]
fn should_parse_a_retiring_entry_that_is_not_a_get_started_pick() {
    let retiring_entry = parse_catalog_entry(
        entry_document_with(&[("role", json!("chat")), ("retiring", json!(true))]),
        0,
        &mut empty_normalized_huggingface_ids(),
        &mut empty_get_started_role_names(),
    )
    .expect("a retiring chat entry should parse");

    assert!(retiring_entry.retiring());
    assert!(!retiring_entry.get_started());
}

#[test]
fn should_reject_a_retiring_get_started_pick() {
    let retiring_pick = entry_document_with(&[
        ("role", json!("chat")),
        ("get_started", json!(true)),
        ("retiring", json!(true)),
    ]);

    assert!(matches!(
        validate_get_started_pick(&retiring_pick, 4, &mut empty_get_started_role_names()),
        Err(DownloadCatalogError::InvalidGetStartedPick { entry_index: 4 })
    ));
}

#[test]
fn should_refuse_a_retiring_flag_that_is_not_a_boolean() {
    let mut catalog_entry = baseline_entry_document();
    catalog_entry["retiring"] = json!("soon");

    let document = serde_json::from_value::<DownloadCatalogEntryDocument>(catalog_entry);

    assert!(document.is_err());
}

#[test]
fn should_accept_one_get_started_pick_per_role() {
    let mut get_started_role_names = empty_get_started_role_names();
    let chat_pick = entry_document_with(&[("role", json!("chat")), ("get_started", json!(true))]);
    let coding_pick =
        entry_document_with(&[("role", json!("coding")), ("get_started", json!(true))]);
    let chat_non_pick = entry_document_with(&[("role", json!("chat"))]);

    assert!(validate_get_started_pick(&chat_pick, 0, &mut get_started_role_names).is_ok());
    assert!(validate_get_started_pick(&coding_pick, 1, &mut get_started_role_names).is_ok());
    assert!(validate_get_started_pick(&chat_non_pick, 2, &mut get_started_role_names).is_ok());
}

#[test]
fn should_reject_two_get_started_picks_for_one_role() {
    let mut get_started_role_names = empty_get_started_role_names();
    let first_chat_pick =
        entry_document_with(&[("role", json!("chat")), ("get_started", json!(true))]);
    let second_chat_pick =
        entry_document_with(&[("role", json!("chat")), ("get_started", json!(true))]);
    assert!(validate_get_started_pick(&first_chat_pick, 0, &mut get_started_role_names).is_ok());

    assert!(matches!(
        validate_get_started_pick(&second_chat_pick, 1, &mut get_started_role_names),
        Err(DownloadCatalogError::DuplicateGetStartedPick { role: "chat" })
    ));
}

#[test]
fn should_reject_an_experimental_helper_or_decision_get_started_pick() {
    let mut get_started_role_names = empty_get_started_role_names();
    let experimental_pick = entry_document_with(&[
        ("role", json!("chat")),
        ("get_started", json!(true)),
        ("experimental", json!(true)),
    ]);
    let helper_pick =
        entry_document_with(&[("role", json!("helper")), ("get_started", json!(true))]);

    assert!(matches!(
        validate_get_started_pick(&experimental_pick, 3, &mut get_started_role_names),
        Err(DownloadCatalogError::InvalidGetStartedPick { entry_index: 3 })
    ));
    assert!(matches!(
        validate_get_started_pick(&helper_pick, 5, &mut get_started_role_names),
        Err(DownloadCatalogError::InvalidGetStartedPick { entry_index: 5 })
    ));
    let decision_pick =
        entry_document_with(&[("role", json!("decision")), ("get_started", json!(true))]);
    assert!(matches!(
        validate_get_started_pick(&decision_pick, 6, &mut get_started_role_names),
        Err(DownloadCatalogError::InvalidGetStartedPick { entry_index: 6 })
    ));
}
