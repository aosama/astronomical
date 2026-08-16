use std::fs;
use std::path::Path;

mod classified_artifacts;
mod family_classification;
mod package_structure;
mod qwen3_5;
mod traversal;

/// Writes the smallest config document needed to exercise family discovery.
fn write_minimal_model_config(
    model_directory: &Path,
    model_type: &str,
    maximum_position_embeddings: u32,
) {
    let model_config_json = serde_json::json!({
        "model_type": model_type,
        "text_config": { "max_position_embeddings": maximum_position_embeddings },
    });
    fs::write(
        model_directory.join("config.json"),
        model_config_json.to_string(),
    )
    .expect("model config should be written");
}

/// Writes common indexed-checkpoint files used by synthetic discovery fixtures.
fn write_required_model_files(model_directory: &Path) {
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{}}"#,
    )
    .expect("safetensors index should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
}

/// Runs discovery against one temporary configured root using the standard test output limit.
fn discover_configured_models(
    temporary_directory: &tempfile::TempDir,
) -> Vec<astronomical_config::ModelDiscoveryDirectoryScan> {
    let configured_model_directories = vec![temporary_directory.path().to_path_buf()];
    astronomical_config::discover_models(&configured_model_directories, 20_480)
        .expect("model discovery should complete")
}
