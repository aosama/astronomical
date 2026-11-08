//! Classification of `models` entries against installed models (#1925).

use std::path::PathBuf;

use crate::{
    AstronomicalConfig, AstronomicalInstancePaths, AstronomicalRuntimeInstance,
    ChatModelCapabilities, DiscoveredModel, EmbeddingModelCapabilities,
    ImageGenerationCapabilities, ModelCapabilities, ModelConfigEntryOutcome, ModelFamily,
    SettingsFreeModelKind,
};

fn config_with_entries(configured_model_ids: &[&str]) -> AstronomicalConfig {
    let mut models_document = serde_json::Map::new();
    for configured_model_id in configured_model_ids {
        models_document.insert(
            (*configured_model_id).to_owned(),
            serde_json::json!({"generation_defaults": {"temperature": 0.5}}),
        );
    }
    let config_document = serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {"model_directories": []},
        "models": models_document,
    });
    AstronomicalConfig::load_v1_bytes(
        AstronomicalInstancePaths::for_state_directory(
            PathBuf::from("/fictional/state"),
            AstronomicalRuntimeInstance::Development,
        ),
        &serde_json::to_vec(&config_document).expect("the config should serialize"),
    )
    .expect("the config should load")
}

fn discovered_model(
    model_id: &str,
    provider_model_id: Option<&str>,
    capabilities: ModelCapabilities,
) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: provider_model_id.map(str::to_owned),
        model_family: ModelFamily::Qwen3_5,
        revision: "fictional-revision".to_owned(),
        model_directory: PathBuf::from("/fictional/models").join(model_id),
        capabilities,
        license: None,
        model_size_bytes: 1,
    }
}

fn chat_model(model_id: &str, provider_model_id: Option<&str>) -> DiscoveredModel {
    discovered_model(
        model_id,
        provider_model_id,
        ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 32_768,
            max_input_tokens: 32_767,
            max_output_tokens: 8_192,
            supports_vision: false,
            supports_reasoning: false,
            supports_tool_calls: false,
        }),
    )
}

fn outcomes(
    configured_model_ids: &[&str],
    discovered_models: &[DiscoveredModel],
) -> Vec<(String, ModelConfigEntryOutcome)> {
    let mut entry_outcomes = Vec::new();
    for entry_report in
        config_with_entries(configured_model_ids).model_config_entry_reports(discovered_models)
    {
        entry_outcomes.push((entry_report.configured_model_id, entry_report.outcome));
    }
    entry_outcomes
}

fn applies(model_id: &str) -> ModelConfigEntryOutcome {
    ModelConfigEntryOutcome::Applies {
        model_id: model_id.to_owned(),
    }
}

#[test]
fn should_apply_a_short_name_entry_and_a_full_id_entry_with_matching_provenance() {
    let installed_models = [
        chat_model("short", None),
        chat_model("tracked", Some("org/tracked")),
    ];

    let entry_outcomes = outcomes(&["org/tracked", "short"], &installed_models);

    assert_eq!(
        entry_outcomes,
        [
            ("org/tracked".to_owned(), applies("tracked")),
            ("short".to_owned(), applies("short")),
        ]
    );
}

#[test]
fn should_say_the_short_name_entry_wins_over_a_full_id_entry() {
    let installed_models = [chat_model("model", Some("org/model"))];

    let entry_outcomes = outcomes(&["model", "org/model"], &installed_models);

    assert_eq!(entry_outcomes[0].1, applies("model"));
    assert_eq!(
        entry_outcomes[1].1,
        ModelConfigEntryOutcome::ConfiguredByAnotherEntry {
            model_id: "model".to_owned(),
            applied_entry_id: "model".to_owned(),
        }
    );
}

#[test]
fn should_report_another_organisation_even_when_the_right_entry_also_exists() {
    let installed_models = [chat_model("model", Some("right-org/model"))];

    let entry_outcomes = outcomes(&["right-org/model", "wrong-org/model"], &installed_models);

    assert_eq!(entry_outcomes[0].1, applies("model"));
    assert_eq!(
        entry_outcomes[1].1,
        ModelConfigEntryOutcome::DifferentOrganisation {
            model_id: "model".to_owned(),
            installed_provider_model_id: "right-org/model".to_owned(),
        }
    );
}

#[test]
fn should_use_the_first_full_id_entry_for_a_model_without_provenance() {
    let installed_models = [chat_model("model", None)];

    let entry_outcomes = outcomes(&["a-org/model", "b-org/model"], &installed_models);

    assert_eq!(entry_outcomes[0].1, applies("model"));
    assert_eq!(
        entry_outcomes[1].1,
        ModelConfigEntryOutcome::ConfiguredByAnotherEntry {
            model_id: "model".to_owned(),
            applied_entry_id: "a-org/model".to_owned(),
        }
    );
}

#[test]
fn should_not_count_an_entry_for_a_picture_or_embedding_model_as_applying() {
    let installed_models = [
        discovered_model(
            "pictures",
            None,
            ModelCapabilities::ImageGeneration(ImageGenerationCapabilities {
                supports_text_to_image: true,
                supports_image_editing: false,
                supports_multiple_reference_images: false,
                default_steps: 4,
                minimum_dimension_pixels: 256,
                maximum_dimension_pixels: 1_024,
                dimension_multiple_pixels: 16,
            }),
        ),
        discovered_model(
            "search",
            None,
            ModelCapabilities::Embeddings(EmbeddingModelCapabilities {
                vector_width: 768,
                max_input_tokens: 512,
            }),
        ),
    ];

    let entry_reports =
        config_with_entries(&["pictures", "search"]).model_config_entry_reports(&installed_models);

    assert!(!entry_reports[0].applies());
    assert_eq!(
        entry_reports[0].outcome,
        ModelConfigEntryOutcome::ModelTakesNoSettings {
            model_id: "pictures".to_owned(),
            model_kind: SettingsFreeModelKind::ImageGeneration,
        }
    );
    assert_eq!(
        entry_reports[1].outcome,
        ModelConfigEntryOutcome::ModelTakesNoSettings {
            model_id: "search".to_owned(),
            model_kind: SettingsFreeModelKind::Embeddings,
        }
    );
}

#[test]
fn should_report_entries_that_match_nothing_including_malformed_keys() {
    let installed_models = [chat_model("model", None)];

    let entry_outcomes = outcomes(&["org/", "org/other", "other"], &installed_models);

    for (_, outcome) in entry_outcomes {
        assert_eq!(outcome, ModelConfigEntryOutcome::MatchesNoInstalledModel);
    }
}

#[test]
fn should_report_nothing_without_entries_and_no_match_without_models() {
    assert!(outcomes(&[], &[chat_model("model", None)]).is_empty());
    assert_eq!(
        outcomes(&["org/model"], &[]),
        [(
            "org/model".to_owned(),
            ModelConfigEntryOutcome::MatchesNoInstalledModel
        )]
    );
}
