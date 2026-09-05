//! Shallow discovery contracts for executable ModernBERT embedding artifacts.

use std::fs;

use astronomical_config::{EmbeddingModelCapabilities, ModelCapabilities};

use super::discover_configured_models;

/// Writes the minimal ModernBERT embedding artifact accepted by shallow discovery.
fn write_modernbert_model_files(model_directory: &std::path::Path) {
    let embedding_config_json = serde_json::json!({
        "model_type": "modernbert",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "max_position_embeddings": 8_192,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160_000.0,
        "local_rope_theta": 10_000.0,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 50_283,
        "quantization": { "group_size": 64, "bits": 8 },
    });
    fs::write(
        model_directory.join("config.json"),
        embedding_config_json.to_string(),
    )
    .expect("embedding config should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
    fs::write(
        model_directory.join("model.safetensors"),
        b"fictional-weights",
    )
    .expect("safetensors should be written");
}

#[test]
fn should_discover_an_executable_modernbert_embedding_model() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory
        .path()
        .join("nomicai-modernbert-embed-base-8bit");
    fs::create_dir(&model_directory).expect("model directory should be created");
    write_modernbert_model_files(&model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_models = directory_scans
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect::<Vec<_>>();

    assert_eq!(discovered_models.len(), 1, "ModernBERT must be discovered");
    let discovered_model = &discovered_models[0];
    assert_eq!(
        discovered_model.model_id,
        "nomicai-modernbert-embed-base-8bit"
    );
    assert_eq!(
        discovered_model.model_family,
        astronomical_config::ModelFamily::ModernBert
    );
    match &discovered_model.capabilities {
        ModelCapabilities::Embeddings(EmbeddingModelCapabilities {
            vector_width,
            max_input_tokens,
        }) => {
            assert_eq!(*vector_width, 768);
            assert_eq!(*max_input_tokens, 8_192);
        }
        other_capabilities => panic!("expected embeddings capability, got {other_capabilities:?}"),
    }
}

#[test]
fn should_reject_a_modernbert_model_without_tokenizer_or_weights() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("modernbert-incomplete");
    fs::create_dir(&model_directory).expect("model directory should be created");
    let embedding_config_json = serde_json::json!({
        "model_type": "modernbert",
        "hidden_size": 768,
        "max_position_embeddings": 8_192,
        "local_attention": 128,
        "quantization": { "group_size": 64, "bits": 8 },
    });
    fs::write(
        model_directory.join("config.json"),
        embedding_config_json.to_string(),
    )
    .expect("embedding config should be written");

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_models = directory_scans
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect::<Vec<_>>();

    assert!(
        discovered_models.is_empty(),
        "incomplete ModernBERT artifacts must stay undiscoverable"
    );
}

#[test]
fn should_reject_unsupported_modernbert_quantization_widths() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("modernbert-4bit");
    fs::create_dir(&model_directory).expect("model directory should be created");
    let embedding_config_json = serde_json::json!({
        "model_type": "modernbert",
        "hidden_size": 768,
        "max_position_embeddings": 8_192,
        "local_attention": 128,
        "quantization": { "group_size": 64, "bits": 4 },
    });
    fs::write(
        model_directory.join("config.json"),
        embedding_config_json.to_string(),
    )
    .expect("embedding config should be written");
    fs::write(model_directory.join("tokenizer.json"), r#"{"version":1}"#)
        .expect("tokenizer should be written");
    fs::write(
        model_directory.join("model.safetensors"),
        b"fictional-weights",
    )
    .expect("safetensors should be written");

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_models = directory_scans
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect::<Vec<_>>();

    assert!(
        discovered_models.is_empty(),
        "only the reviewed 8-bit profile is executable"
    );
}
