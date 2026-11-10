//! Shallow discovery contracts for Clef decision-model folders: a Qwen3.5
//! checkpoint whose joint schema head marks it as a decision model.

use std::fs;
use std::path::Path;

use astronomical_config::{
    DecisionModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily,
};

const BACKBONE_SHARD_NAME: &str = "model-00001-of-00001.safetensors";
const CLEF_HEAD_SETTINGS_JSON: &str = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}"#;

/// Writes the files an MLX Clef build holds, with fictional bytes.
fn write_clef_folder(model_directory: &Path, head_settings_json: &str) {
    fs::create_dir_all(model_directory).expect("model directory should be created");
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5","text_config":{"hidden_size":4096}}"#,
    )
    .expect("config should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":1},"weight_map":{}}"#,
    )
    .expect("index should be written");
    fs::write(
        model_directory.join("joint_head_config.json"),
        head_settings_json,
    )
    .expect("head settings should be written");
    fs::write(
        model_directory.join("joint_head.safetensors"),
        b"joint-head",
    )
    .expect("head should be written");
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

fn discovered_models_in(temporary_directory: &tempfile::TempDir) -> Vec<DiscoveredModel> {
    super::discover_configured_models(temporary_directory)
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect()
}

#[test]
fn should_discover_a_clef_folder_as_a_decision_model_with_its_sequence_limit() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_clef_folder(
        &temporary_directory.path().join("clef-flash-8bit"),
        CLEF_HEAD_SETTINGS_JSON,
    );

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models.len(), 1, "Clef must be discovered once");
    assert_eq!(discovered_models[0].model_id, "clef-flash-8bit");
    assert_eq!(discovered_models[0].model_family, ModelFamily::Clef);
    assert_eq!(
        discovered_models[0].capabilities,
        ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: 16_384,
        }),
        "Clef reads at most 16,384 tokens for the whole request"
    );
}

#[test]
fn should_count_the_head_and_backbone_in_the_model_size() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("clef-flash-8bit");
    write_clef_folder(&model_directory, CLEF_HEAD_SETTINGS_JSON);
    let expected_size_bytes = fs::metadata(model_directory.join(BACKBONE_SHARD_NAME))
        .expect("shard metadata should be readable")
        .len()
        + fs::metadata(model_directory.join("joint_head.safetensors"))
            .expect("head metadata should be readable")
            .len();

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models[0].model_size_bytes, expected_size_bytes);
}

#[test]
fn should_not_read_a_folder_without_the_head_weights_as_clef() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("clef-flash-8bit");
    write_clef_folder(&model_directory, CLEF_HEAD_SETTINGS_JSON);
    fs::remove_file(model_directory.join("joint_head.safetensors"))
        .expect("head should be removable");

    assert!(
        discovered_models_in(&temporary_directory)
            .iter()
            .all(|discovered_model| discovered_model.model_family != ModelFamily::Clef),
        "the head settings alone do not make a Clef model"
    );
}

#[test]
fn should_not_read_head_settings_without_a_hidden_size_as_clef() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_clef_folder(
        &temporary_directory.path().join("clef-flash-8bit"),
        r#"{"width": 1024}"#,
    );

    assert!(
        discovered_models_in(&temporary_directory)
            .iter()
            .all(|discovered_model| discovered_model.model_family != ModelFamily::Clef)
    );
}

#[test]
fn should_skip_a_clef_folder_holding_only_its_head() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("clef-flash-8bit");
    write_clef_folder(&model_directory, CLEF_HEAD_SETTINGS_JSON);
    fs::remove_file(model_directory.join(BACKBONE_SHARD_NAME)).expect("shard should be removable");

    assert!(
        discovered_models_in(&temporary_directory).is_empty(),
        "a head without backbone weights is an incomplete download"
    );
}
