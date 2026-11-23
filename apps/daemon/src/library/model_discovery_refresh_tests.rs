use config::DiscoveredModel;

use crate::library::discovered_model_test_support::discovered_chat_model;

use super::{RescanPublicationOrder, model_library_change, rescan_publication_order};

fn discovered_model(model_id: &str, model_directory: &str) -> DiscoveredModel {
    discovered_chat_model(model_id, None, "fictional-revision", model_directory)
}

#[test]
fn should_find_no_change_when_the_library_is_the_same() {
    let library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    assert!(model_library_change(&library, &library).is_empty());
}

#[test]
fn should_report_a_folder_copied_in_as_added() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    let mut rescanned_library = live_library.clone();
    rescanned_library.push(discovered_model(
        "example-ornith",
        "/fictional/models/org/example-ornith",
    ));
    let library_change = model_library_change(&live_library, &rescanned_library);
    assert_eq!(
        library_change.added_model_ids,
        vec!["example-ornith".to_owned()]
    );
    assert!(library_change.removed_model_ids.is_empty());
}

#[test]
fn should_report_a_moved_folder_as_changed_not_removed() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org-a/example-qwen",
    )];
    let rescanned_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org-b/example-qwen",
    )];
    let library_change = model_library_change(&live_library, &rescanned_library);
    assert_eq!(
        library_change.changed_model_ids,
        vec!["example-qwen".to_owned()]
    );
    assert!(
        library_change.added_model_ids.is_empty() && library_change.removed_model_ids.is_empty()
    );
}

#[test]
fn should_report_a_deleted_folder_as_removed() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    let library_change = model_library_change(&live_library, &[]);
    assert_eq!(
        library_change.removed_model_ids,
        vec!["example-qwen".to_owned()]
    );
}

#[test]
fn should_bridge_policies_when_a_rename_both_adds_and_removes_a_model() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    let rescanned_library = vec![discovered_model(
        "example-qwen-renamed",
        "/fictional/models/org/example-qwen-renamed",
    )];
    let library_change = model_library_change(&live_library, &rescanned_library);
    assert_eq!(
        library_change.added_model_ids,
        vec!["example-qwen-renamed".to_owned()]
    );
    assert_eq!(
        library_change.removed_model_ids,
        vec!["example-qwen".to_owned()]
    );
    assert_eq!(
        rescan_publication_order(&library_change),
        RescanPublicationOrder::BridgingPoliciesThenListThenPolicies
    );
}

#[test]
fn should_give_a_gained_model_its_policy_before_listing_it() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    let mut rescanned_library = live_library.clone();
    rescanned_library.push(discovered_model(
        "example-ornith",
        "/fictional/models/org/example-ornith",
    ));
    assert_eq!(
        rescan_publication_order(&model_library_change(&live_library, &rescanned_library)),
        RescanPublicationOrder::PoliciesThenList
    );
}

#[test]
fn should_unlist_a_lost_model_before_dropping_its_policy() {
    let live_library = vec![discovered_model(
        "example-qwen",
        "/fictional/models/org/example-qwen",
    )];
    assert_eq!(
        rescan_publication_order(&model_library_change(&live_library, &[])),
        RescanPublicationOrder::ListThenPolicies
    );
}
