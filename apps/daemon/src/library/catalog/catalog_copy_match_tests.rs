use std::path::PathBuf;

use config::{ChatModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily};

use super::catalog_copy_match::find_discovered_copy;

const CATALOG_REPOSITORY: &str = "example-org/Example-Chat-4bit";
const CATALOG_REVISION: &str = "1111111111111111111111111111111111111111";
const OTHER_REVISION: &str = "2222222222222222222222222222222222222222";

fn discovered_copy(
    provider_model_id: Option<&str>,
    revision: &str,
    model_directory: &str,
) -> DiscoveredModel {
    DiscoveredModel {
        model_id: "Example-Chat-4bit".to_owned(),
        provider_model_id: provider_model_id.map(str::to_owned),
        model_family: ModelFamily::Qwen3_5,
        revision: revision.to_owned(),
        model_directory: PathBuf::from(model_directory),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 65_536,
            max_input_tokens: 65_535,
            max_output_tokens: 20_480,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1_000,
    }
}

#[test]
fn should_match_a_downloaded_copy_at_the_catalog_revision() {
    let discovered_models = [discovered_copy(
        Some(CATALOG_REPOSITORY),
        CATALOG_REVISION,
        "/fictional/models/example-org/Example-Chat-4bit",
    )];

    let matched_copy =
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models);

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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models);

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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models).is_some(),
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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models).is_some()
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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models).is_none(),
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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models).is_none()
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
        find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &discovered_models).is_none()
    );
    assert!(find_discovered_copy(CATALOG_REPOSITORY, CATALOG_REVISION, &[]).is_none());
}
