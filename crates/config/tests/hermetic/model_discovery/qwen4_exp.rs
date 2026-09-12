//! Recognition contracts for the `qwen4_exp` family (Qwen 3.8 Flash).
//!
//! The user-visible outcome under test: a Qwen 3.8 Flash directory is
//! identified as a known family everywhere a family name matters, never shows
//! up as an advertised or downloadable model, and refuses execution with an
//! honest reason instead of disappearing or pretending to load.

use std::fs;

use astronomical_config::{
    ModelFamily, Qwen4ExpConfigurationSummary, classify_model_directory, describe_configuration,
    discover_classified_model_artifacts,
};

use super::{discover_configured_models, write_required_model_files};

const CONFIG_JSON: &str = r#"{
    "model_type": "qwen4_exp",
    "text_config": {
        "num_hidden_layers": 48,
        "max_position_embeddings": 262144,
        "num_experts": 288
    }
}"#;

fn write_family_fixture(model_directory: &std::path::Path, model_type: &str) {
    fs::create_dir_all(model_directory).expect("model directory should be created");
    let config = CONFIG_JSON.replace("qwen4_exp", model_type);
    fs::write(model_directory.join("config.json"), config).expect("model config should be written");
    write_required_model_files(model_directory);
}

#[test]
fn should_classify_the_conditional_generation_wrapper_without_advertising_it() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Qwen3.8-Flash-Fixture");
    write_family_fixture(&model_directory, "qwen4_exp");

    assert_eq!(
        classify_model_directory(&model_directory)
            .expect("qwen4_exp classification should complete"),
        Some(ModelFamily::Qwen4Exp)
    );
    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty(),
        "recognized qwen4_exp artifacts must stay unpublished until serving exists"
    );
    let classified_artifacts =
        discover_classified_model_artifacts(&[temporary_directory.path().to_path_buf()])
            .expect("classified qwen4_exp discovery should complete");
    assert_eq!(classified_artifacts.len(), 1);
    assert_eq!(classified_artifacts[0].model_family, ModelFamily::Qwen4Exp);
}

#[test]
fn should_classify_a_text_only_distribution_of_the_same_family() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory
        .path()
        .join("Qwen3.8-Flash-Text-Fixture");
    write_family_fixture(&model_directory, "qwen4_exp_text");

    assert_eq!(
        classify_model_directory(&model_directory)
            .expect("qwen4_exp_text classification should complete"),
        Some(ModelFamily::Qwen4Exp)
    );
}

#[test]
fn should_summarize_the_text_configuration_for_diagnostics() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Qwen3.8-Flash-Fixture");
    write_family_fixture(&model_directory, "qwen4_exp");
    let config_bytes =
        fs::read(model_directory.join("config.json")).expect("config should be readable");
    let config_value: serde_json::Value =
        serde_json::from_slice(&config_bytes).expect("config should parse");

    assert_eq!(
        describe_configuration(&config_value),
        Some(Qwen4ExpConfigurationSummary {
            decoder_layers: 48,
            context_window_tokens: 262_144,
            routed_experts: 288,
        })
    );
}

#[test]
fn should_refuse_a_summary_from_a_document_without_the_text_configuration() {
    let flat_document = serde_json::json!({
        "model_type": "qwen4_exp",
        "num_hidden_layers": 48,
        "max_position_embeddings": 262_144,
        "num_experts": 288
    });
    assert_eq!(
        describe_configuration(&flat_document),
        None,
        "a root-level layout without the nested text configuration is not a summary"
    );
}

#[test]
fn should_report_a_malformed_configuration_as_an_error_not_an_unknown_family() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory
        .path()
        .join("Qwen3.8-Flash-Broken-Fixture");
    fs::create_dir_all(&model_directory).expect("model directory should be created");
    fs::write(model_directory.join("config.json"), "{not json")
        .expect("broken config should be written");

    let error = classify_model_directory(&model_directory)
        .expect_err("a malformed configuration must be an error");
    assert!(
        error
            .to_string()
            .contains("failed to parse model config.json"),
        "the error should name the failing document: {error}"
    );
}
