use std::fs;
use std::path::Path;

use astronomical_config::{AstronomicalConfig, ModelFamily};

use crate::hermetic::write_config;

fn write_minimal_model_config(
    model_directory: &Path,
    model_type: &str,
    maximum_position_embeddings: u32,
) {
    let model_config_json = serde_json::json!({
        "model_type": model_type,
        "text_config": { "max_position_embeddings": maximum_position_embeddings },
        "quantization_config": { "quant_method": "optiq", "per_layer": [] }
    });
    fs::write(
        model_directory.join("config.json"),
        model_config_json.to_string(),
    )
    .expect("model config should be written");
}

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

fn write_dense_qwen3_5_vision_model_files(model_directory: &Path) {
    let dense_model_config_json = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": { "max_position_embeddings": 131_072 },
        "vision_config": { "model_type": "qwen3_5_vision" },
    });
    fs::write(
        model_directory.join("config.json"),
        dense_model_config_json.to_string(),
    )
    .expect("dense model config should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
}

fn write_embedded_vision_model_files(model_directory: &Path) {
    fs::write(model_directory.join("model-00001.safetensors"), [])
        .expect("embedded vision shard should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{"vision_tower.patch_embed.proj.weight":"model-00001.safetensors"}}"#,
    )
    .expect("embedded vision index should be written");
}

fn write_separate_vision_model_files(model_directory: &Path) {
    fs::create_dir_all(model_directory.join("optiq"))
        .expect("vision sidecar directory should be created");
    fs::write(model_directory.join("model-00001.safetensors"), [])
        .expect("language model shard should be written");
    fs::write(model_directory.join("optiq/optiq_vision.safetensors"), [])
        .expect("vision sidecar should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{"language_model.model.embed_tokens.weight":"model-00001.safetensors","vision_tower.patch_embed.proj.weight":"optiq/optiq_vision.safetensors"}}"#,
    )
    .expect("vision sidecar index should be written");
}

fn discover_configured_models(
    temporary_directory: &tempfile::TempDir,
) -> Vec<astronomical_config::ModelDiscoveryDirectoryScan> {
    let configured_model_directories = vec![temporary_directory.path().to_path_buf()];
    astronomical_config::discover_models(&configured_model_directories, 20_480)
        .expect("model discovery should complete")
}

#[test]
fn should_discover_supported_text_and_vision_models() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let text_model_directory = temporary_directory.path().join("TextModel-OptiQ-4bit");
    let vision_model_directory = temporary_directory.path().join("VisionModel-OptiQ-4bit");
    fs::create_dir_all(&text_model_directory).expect("text model directory should be created");
    fs::create_dir_all(&vision_model_directory).expect("vision model directory should be created");
    write_minimal_model_config(&text_model_directory, "qwen3_5_moe", 262_144);
    write_minimal_model_config(&vision_model_directory, "qwen3_5_moe_vision", 262_144);
    write_required_model_files(&text_model_directory);
    write_required_model_files(&vision_model_directory);
    write_embedded_vision_model_files(&vision_model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);

    assert_eq!(directory_scans.len(), 1);
    assert_eq!(directory_scans[0].discovered_models.len(), 2);
    assert!(
        directory_scans[0]
            .discovered_models
            .iter()
            .any(|discovered_model| discovered_model.model_id == "TextModel-OptiQ-4bit")
    );
    assert!(
        directory_scans[0]
            .discovered_models
            .iter()
            .any(|discovered_model| discovered_model.has_vision)
    );
}

#[test]
fn should_discover_a_dense_qwen3_5_model_as_text_only_despite_vision_metadata() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let dense_model_directory = temporary_directory.path().join("DenseQwen3_5-OptiQ-4bit");
    fs::create_dir_all(dense_model_directory.join("optiq"))
        .expect("dense model and optional sidecar directory should be created");
    let dense_model_config_json = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": { "max_position_embeddings": 131_072 },
        "vision_config": { "model_type": "qwen3_5" },
    });
    fs::write(
        dense_model_directory.join("config.json"),
        dense_model_config_json.to_string(),
    )
    .expect("dense model config should be written");
    write_required_model_files(&dense_model_directory);
    fs::write(
        dense_model_directory.join("optiq/optiq_vision.safetensors"),
        [],
    )
    .expect("optional vision sidecar should be written");
    fs::write(dense_model_directory.join("optiq/mtp.safetensors"), [])
        .expect("optional MTP sidecar should be written");

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == "DenseQwen3_5-OptiQ-4bit")
        .expect("dense Qwen3.5 model should be discovered");

    assert!(!discovered_model.has_vision);
    assert_eq!(discovered_model.context_window, 131_072);
}

#[test]
fn should_allow_a_missing_mtp_only_file_with_an_arbitrary_name() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("MoeQwen3_5-4bit");
    fs::create_dir_all(&model_directory).expect("model directory should be created");
    write_minimal_model_config(&model_directory, "qwen3_5_moe", 131_072);
    fs::write(model_directory.join("tokenizer.json"), "{}").expect("tokenizer should be written");
    fs::write(model_directory.join("model-00001.safetensors"), [])
        .expect("target model shard should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":0},"weight_map":{"language_model.model.embed_tokens.weight":"model-00001.safetensors","language_model.mtp.fc.weight":"predictor-weights.safetensors"}}"#,
    )
    .expect("model index should be written");

    let directory_scans = discover_configured_models(&temporary_directory);

    assert_eq!(directory_scans[0].discovered_models.len(), 1);
}

#[test]
fn should_discover_a_dense_qwen3_5_model_with_embedded_vision_as_image_capable() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let dense_model_directory = temporary_directory
        .path()
        .join("DenseQwen3_5Vision-OptiQ-4bit");
    fs::create_dir_all(&dense_model_directory).expect("dense model directory should be created");
    write_dense_qwen3_5_vision_model_files(&dense_model_directory);
    write_embedded_vision_model_files(&dense_model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == "DenseQwen3_5Vision-OptiQ-4bit")
        .expect("dense Qwen3.5 vision model should be discovered");

    assert!(discovered_model.has_vision);
    assert_eq!(discovered_model.context_window, 131_072);
}

#[test]
fn should_discover_a_dense_qwen3_5_model_with_a_vision_sidecar_as_image_capable() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let dense_model_directory = temporary_directory
        .path()
        .join("DenseQwen3_5Sidecar-OptiQ-4bit");
    fs::create_dir_all(&dense_model_directory).expect("dense model directory should be created");
    write_dense_qwen3_5_vision_model_files(&dense_model_directory);
    write_separate_vision_model_files(&dense_model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == "DenseQwen3_5Sidecar-OptiQ-4bit")
        .expect("dense Qwen3.5 sidecar model should be discovered");

    assert!(discovered_model.has_vision);
    assert_eq!(discovered_model.context_window, 131_072);
}

#[test]
fn should_skip_a_dense_qwen3_5_model_with_an_indexed_but_missing_vision_sidecar() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let dense_model_directory = temporary_directory
        .path()
        .join("DenseQwen3_5MissingSidecar-OptiQ-4bit");
    fs::create_dir_all(&dense_model_directory).expect("dense model directory should be created");
    write_dense_qwen3_5_vision_model_files(&dense_model_directory);
    write_separate_vision_model_files(&dense_model_directory);
    fs::remove_file(dense_model_directory.join("optiq/optiq_vision.safetensors"))
        .expect("vision sidecar should be removed");

    let directory_scans = discover_configured_models(&temporary_directory);

    assert!(directory_scans[0].discovered_models.is_empty());
}

#[test]
fn should_skip_unsupported_and_incomplete_model_directories() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let unsupported_model_directory = temporary_directory.path().join("UnsupportedModel");
    let incomplete_model_directory = temporary_directory.path().join("IncompleteModel");
    fs::create_dir_all(&unsupported_model_directory).expect("model directory should be created");
    fs::create_dir_all(&incomplete_model_directory).expect("model directory should be created");
    write_minimal_model_config(&unsupported_model_directory, "llama", 4_096);
    write_required_model_files(&unsupported_model_directory);
    write_minimal_model_config(&incomplete_model_directory, "qwen3_5_moe", 262_144);

    let directory_scans = discover_configured_models(&temporary_directory);

    assert!(directory_scans[0].discovered_models.is_empty());
}

#[test]
fn should_classify_deepseek_v4_without_discovering_it_as_executable() {
    assert_eq!(
        ModelFamily::from_model_type(Some("deepseek_v4")),
        Some(ModelFamily::DeepSeekV4)
    );

    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let deepseek_model_directory = temporary_directory.path().join("DeepSeek-V4-Flash-0731");
    fs::create_dir_all(&deepseek_model_directory)
        .expect("DeepSeek model directory should be created");
    write_minimal_model_config(&deepseek_model_directory, "deepseek_v4", 262_144);
    write_required_model_files(&deepseek_model_directory);

    let directory_scans =
        astronomical_config::discover_models(&[temporary_directory.path().to_path_buf()], 20_480)
            .expect("model discovery should complete");

    assert!(directory_scans[0].discovered_models.is_empty());
}

#[test]
fn should_measure_unique_model_shard_bytes_during_discovery() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("MeasuredModel");
    fs::create_dir_all(&model_directory).expect("model directory should be created");
    write_minimal_model_config(&model_directory, "qwen3_5_moe", 262_144);
    fs::write(model_directory.join("tokenizer.json"), "{}").expect("tokenizer should be written");
    fs::write(
        model_directory.join("model-00001.safetensors"),
        vec![0_u8; 123],
    )
    .expect("model shard should be written");
    fs::create_dir_all(model_directory.join("optiq")).expect("OptiQ directory should be written");
    fs::write(
        model_directory.join("optiq/optiq_vision.safetensors"),
        vec![0_u8; 77],
    )
    .expect("vision sidecar should be written");
    fs::write(model_directory.join("notes.txt"), vec![0_u8; 50])
        .expect("unrelated file should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"weight_map":{"first":"model-00001.safetensors","second":"model-00001.safetensors"}}"#,
    )
    .expect("safetensors index should be written");

    let directory_scans = discover_configured_models(&temporary_directory);

    assert_eq!(
        directory_scans[0].discovered_models[0].model_size_bytes,
        200
    );
}

#[test]
fn should_resolve_model_directory_paths_from_the_configuration_array() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let config_directory = temporary_home_directory.path().join(".astronomical");
    let configured_model_directory = temporary_home_directory
        .path()
        .join("models/ConfiguredModel");
    fs::create_dir_all(&config_directory).expect("config directory should be created");
    fs::create_dir_all(&configured_model_directory).expect("model directory should be created");
    let config_json = serde_json::json!({
        "model_directories": [configured_model_directory],
        "chunking": {"prefill_size_optimizer_enabled": true}
    });
    fs::write(
        config_directory.join("config.json"),
        config_json.to_string(),
    )
    .expect("config should be written");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");
    let configured_model_directories = astronomical_config.model_directories();

    assert_eq!(configured_model_directories.len(), 1);
    assert_eq!(configured_model_directories[0], configured_model_directory);
}

#[test]
fn should_resolve_an_exact_model_directory_from_configured_recursive_roots() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let configured_model_root = temporary_home_directory.path().join("models");
    let exact_model_directory = configured_model_root
        .join("nested")
        .join("ExactModel-OptiQ-4bit");
    fs::create_dir_all(&exact_model_directory).expect("model directory should be created");
    write_minimal_model_config(&exact_model_directory, "qwen3_5_moe", 262_144);
    write_required_model_files(&exact_model_directory);
    write_config(
        temporary_home_directory.path(),
        &serde_json::json!({
            "model_directories": [configured_model_root],
            "chunking": { "prefill_size_optimizer_enabled": true },
        })
        .to_string(),
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .find_configured_model_directory_by_id("ExactModel-OptiQ-4bit")
            .expect("configured model discovery should complete"),
        Some(exact_model_directory)
    );
    assert_eq!(
        astronomical_config
            .find_configured_model_directory_by_id("MissingModel-OptiQ-4bit")
            .expect("configured model discovery should complete for a missing model"),
        None
    );
}
