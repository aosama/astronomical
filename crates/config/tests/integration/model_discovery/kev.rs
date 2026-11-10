//! Shallow discovery contracts for Kev decision-model folders: a Qwen3.5
//! backbone whose `kev_decision.json` marks it as a decision model.

use std::fs;
use std::path::Path;

use astronomical_config::{
    DecisionModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily,
};

const BACKBONE_SHARD_NAME: &str = "model.safetensors-00001-of-00001.safetensors";

/// Writes the files `cargo xtask convert-kev` leaves, with fictional bytes.
fn write_kev_folder(model_directory: &Path, kev_settings_json: &str) {
    fs::create_dir_all(model_directory).expect("model directory should be created");
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5","text_config":{"hidden_size":1024}}"#,
    )
    .expect("config should be written");
    fs::write(model_directory.join("kev_decision.json"), kev_settings_json)
        .expect("kev settings should be written");
    fs::write(
        model_directory.join("kev_head.safetensors"),
        b"pointer-head",
    )
    .expect("pointer head should be written");
    fs::write(
        model_directory.join(BACKBONE_SHARD_NAME),
        b"fictional-backbone-weights-longer-than-the-head",
    )
    .expect("backbone shard should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
}

const KEV_SETTINGS_JSON: &str =
    r#"{"model_type":"kev","temperature":2.35,"pointer_dimension":256}"#;

fn discovered_models_in(temporary_directory: &tempfile::TempDir) -> Vec<DiscoveredModel> {
    super::discover_configured_models(temporary_directory)
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect()
}

#[test]
fn should_discover_a_kev_folder_as_a_decision_model_not_a_chat_model() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_kev_folder(
        &temporary_directory.path().join("kev-0.8b-astronomical"),
        KEV_SETTINGS_JSON,
    );

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models.len(), 1, "Kev must be discovered once");
    assert_eq!(discovered_models[0].model_id, "kev-0.8b-astronomical");
    assert_eq!(discovered_models[0].model_family, ModelFamily::Kev);
    assert_eq!(
        discovered_models[0].capabilities,
        ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: 73_728,
        }),
        "a Kev row holds a 65,536-token state and an 8,192-token question"
    );
}

#[test]
fn should_count_every_safetensors_file_in_the_model_size() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("kev-0.8b-astronomical");
    write_kev_folder(&model_directory, KEV_SETTINGS_JSON);
    let expected_size_bytes = fs::metadata(model_directory.join(BACKBONE_SHARD_NAME))
        .expect("shard metadata should be readable")
        .len()
        + fs::metadata(model_directory.join("kev_head.safetensors"))
            .expect("head metadata should be readable")
            .len();

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models[0].model_size_bytes, expected_size_bytes);
}

#[test]
fn should_read_a_folder_whose_marker_names_another_model_type_as_qwen() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_kev_folder(
        &temporary_directory.path().join("not-kev"),
        r#"{"model_type":"something-else"}"#,
    );

    let discovered_models = discovered_models_in(&temporary_directory);

    assert!(
        discovered_models
            .iter()
            .all(|discovered_model| discovered_model.model_family != ModelFamily::Kev),
        "only the kev marker makes a folder a Kev model"
    );
}

#[test]
fn should_skip_a_kev_folder_without_its_pointer_head() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("kev-0.8b-astronomical");
    write_kev_folder(&model_directory, KEV_SETTINGS_JSON);
    fs::remove_file(model_directory.join("kev_head.safetensors"))
        .expect("head should be removable");

    assert!(discovered_models_in(&temporary_directory).is_empty());
}

#[test]
fn should_skip_a_kev_folder_holding_only_its_head() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("kev-0.8b-astronomical");
    write_kev_folder(&model_directory, KEV_SETTINGS_JSON);
    fs::remove_file(model_directory.join(BACKBONE_SHARD_NAME)).expect("shard should be removable");

    assert!(
        discovered_models_in(&temporary_directory).is_empty(),
        "a head without backbone weights is an incomplete download"
    );
}

#[test]
fn should_skip_a_kev_folder_whose_marker_is_not_json() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_kev_folder(
        &temporary_directory.path().join("kev-0.8b-astronomical"),
        "not json",
    );

    assert!(
        discovered_models_in(&temporary_directory)
            .iter()
            .all(|discovered_model| discovered_model.model_family != ModelFamily::Kev)
    );
}
