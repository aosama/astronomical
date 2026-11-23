use config::DiscoveredModel;

use crate::library::discovered_model_test_support::discovered_chat_model;

use super::catalog_copy_match::{find_discovered_copy, find_exact_copy};

const CATALOG_REPOSITORY: &str = "example-org/Example-Chat-4bit";
const CATALOG_REVISION: &str = "1111111111111111111111111111111111111111";
const OTHER_REVISION: &str = "2222222222222222222222222222222222222222";
const FORMER_REPOSITORY: &str = "example-org/Example-Chat-Old-Name";
const RENAMED_ENTRY_NAMES: [&str; 2] = [CATALOG_REPOSITORY, FORMER_REPOSITORY];

/// A copy downloaded before the repository was renamed: its provenance and
/// its folder both carry the former name.
fn copy_under_former_name(provider_model_id: Option<&str>, revision: &str) -> DiscoveredModel {
    discovered_chat_model(
        "Example-Chat-Old-Name",
        provider_model_id,
        revision,
        "/fictional/models/example-org/Example-Chat-Old-Name",
    )
}

fn discovered_copy(
    provider_model_id: Option<&str>,
    revision: &str,
    model_directory: &str,
) -> DiscoveredModel {
    discovered_chat_model(
        "Example-Chat-4bit",
        provider_model_id,
        revision,
        model_directory,
    )
}

#[test]
fn should_match_a_downloaded_copy_at_the_catalog_revision() {
    let discovered_models = [discovered_copy(
        Some(CATALOG_REPOSITORY),
        CATALOG_REVISION,
        "/fictional/models/example-org/Example-Chat-4bit",
    )];

    let matched_copy =
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models);

    assert_eq!(
        matched_copy.map(|copy| copy.revision.as_str()),
        Some(CATALOG_REVISION)
    );
}

#[test]
fn should_prefer_the_exact_revision_over_another_copy_of_the_same_repository() {
    let discovered_models = [
        discovered_copy(
            Some(CATALOG_REPOSITORY),
            OTHER_REVISION,
            "/fictional/cache/models--example-org--Example-Chat-4bit/snapshots/old",
        ),
        discovered_copy(
            Some(CATALOG_REPOSITORY),
            CATALOG_REVISION,
            "/fictional/models/example-org/Example-Chat-4bit",
        ),
    ];

    let matched_copy =
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models);

    assert_eq!(
        matched_copy.map(|copy| copy.revision.as_str()),
        Some(CATALOG_REVISION)
    );
}

#[test]
fn should_match_a_copy_of_the_repository_at_another_revision() {
    let discovered_models = [discovered_copy(
        Some(CATALOG_REPOSITORY),
        OTHER_REVISION,
        "/fictional/models/example-org/Example-Chat-4bit",
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_some(),
        "an older or newer build of the same model is still on this Mac"
    );
}

#[test]
fn should_match_a_hand_copied_folder_without_provenance_by_its_publisher_folder() {
    let discovered_models = [discovered_copy(
        None,
        "derived-from-config",
        "/fictional/models/example-org/Example-Chat-4bit",
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_some()
    );
}

#[test]
fn should_not_match_a_copy_whose_provenance_names_another_repository() {
    let discovered_models = [discovered_copy(
        Some("another-org/Example-Chat-4bit"),
        CATALOG_REVISION,
        "/fictional/models/example-org/Example-Chat-4bit",
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_none(),
        "provenance outranks the folder it sits in"
    );
}

#[test]
fn should_not_match_a_folder_without_provenance_under_another_publisher() {
    let discovered_models = [discovered_copy(
        None,
        "derived-from-config",
        "/fictional/models/another-org/Example-Chat-4bit",
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_none()
    );
}

#[test]
fn should_not_match_a_bare_model_folder_or_an_empty_discovery() {
    let discovered_models = [discovered_copy(
        None,
        "derived-from-config",
        "Example-Chat-4bit",
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_none()
    );
    assert!(find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &[]).is_none());
}

#[test]
fn should_treat_a_copy_downloaded_under_a_former_name_as_installed() {
    let discovered_models = [copy_under_former_name(
        Some(FORMER_REPOSITORY),
        CATALOG_REVISION,
    )];

    let installed_copy =
        find_exact_copy(&RENAMED_ENTRY_NAMES, CATALOG_REVISION, &discovered_models);

    assert_eq!(
        installed_copy.map(|copy| copy.model_id.as_str()),
        Some("Example-Chat-Old-Name"),
        "the old copy keeps the name the engine serves it under"
    );
}

#[test]
fn should_serve_but_not_install_a_former_name_copy_at_another_revision() {
    let discovered_models = [copy_under_former_name(
        Some(FORMER_REPOSITORY),
        OTHER_REVISION,
    )];

    assert!(find_exact_copy(&RENAMED_ENTRY_NAMES, CATALOG_REVISION, &discovered_models).is_none());
    assert!(
        find_discovered_copy(&RENAMED_ENTRY_NAMES, CATALOG_REVISION, &discovered_models).is_some()
    );
}

#[test]
fn should_match_a_hand_copied_folder_spelling_a_former_name() {
    let discovered_models = [copy_under_former_name(None, "derived-from-config")];

    assert!(
        find_discovered_copy(&RENAMED_ENTRY_NAMES, CATALOG_REVISION, &discovered_models).is_some()
    );
}

#[test]
fn should_not_match_a_former_name_copy_when_the_entry_lists_no_former_names() {
    let discovered_models = [copy_under_former_name(
        Some(FORMER_REPOSITORY),
        CATALOG_REVISION,
    )];

    assert!(
        find_discovered_copy(&[CATALOG_REPOSITORY], CATALOG_REVISION, &discovered_models).is_none()
    );
}
