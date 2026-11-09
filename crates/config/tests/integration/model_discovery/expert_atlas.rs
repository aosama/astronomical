//! Discovery of Expert Atlas models: a manifest plus a resident bundle and
//! one record file per expert layer, with no shard index.

use std::fs;
use std::path::Path;

const RESIDENT_BUNDLE_BYTES: usize = 300;
const LAYER_FILE_BYTES: usize = 16_384;

/// Every model discovered under `models_root`.
fn discovered_models_under(models_root: &Path) -> Vec<astronomical_config::DiscoveredModel> {
    astronomical_config::discover_models(&[models_root.to_path_buf()])
        .expect("discovery completes")
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect()
}

/// An Atlas directory whose manifest declares `format_version`.
fn write_atlas_model(model_directory: &Path, format_version: u32, with_resident_bundle: bool) {
    fs::create_dir_all(model_directory.join("experts")).expect("model directory should be created");
    super::write_minimal_model_config(model_directory, "qwen3_5_moe", 262_144);
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
    if with_resident_bundle {
        fs::write(
            model_directory.join("resident.safetensors"),
            vec![0_u8; RESIDENT_BUNDLE_BYTES],
        )
        .expect("resident bundle should be written");
    }
    fs::write(
        model_directory.join("experts/layer-000.atlas"),
        vec![0_u8; LAYER_FILE_BYTES],
    )
    .expect("layer file should be written");
    let atlas_manifest = serde_json::json!({
        "format": "astronomical-expert-atlas",
        "format_version": format_version,
        "source": { "model_id": "organization/model" },
        "resident_files": [],
        "expert_files": [],
        "expert_layers": [],
    });
    fs::write(
        model_directory.join("manifest.json"),
        atlas_manifest.to_string(),
    )
    .expect("manifest should be written");
}

#[test]
fn should_discover_an_atlas_model_and_count_its_layer_files_in_its_size() {
    let models_root = tempfile::tempdir().expect("models root should be created");
    write_atlas_model(
        &models_root
            .path()
            .join("organization/model-astro-atlas-streaming"),
        4,
        true,
    );
    let discovered_models = discovered_models_under(models_root.path());
    let discovered_model = discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == "model-astro-atlas-streaming")
        .expect("the Atlas model is discovered");
    assert_eq!(
        discovered_model.model_size_bytes,
        (RESIDENT_BUNDLE_BYTES + LAYER_FILE_BYTES) as u64
    );
}

#[test]
fn should_not_discover_a_retired_version_3_manifest() {
    let models_root = tempfile::tempdir().expect("models root should be created");
    write_atlas_model(
        &models_root
            .path()
            .join("organization/old-streaming-revision"),
        3,
        true,
    );
    assert!(
        discovered_models_under(models_root.path()).is_empty(),
        "a version 3 manifest is not a servable model"
    );
}

#[test]
fn should_not_discover_an_atlas_without_its_resident_bundle() {
    let models_root = tempfile::tempdir().expect("models root should be created");
    write_atlas_model(
        &models_root.path().join("organization/incomplete-atlas"),
        4,
        false,
    );
    assert!(
        discovered_models_under(models_root.path()).is_empty(),
        "an Atlas missing its resident bundle cannot load"
    );
}

#[test]
fn should_discover_a_stock_model_that_ships_an_unrelated_manifest() {
    let models_root = tempfile::tempdir().expect("models root should be created");
    let model_directory = models_root.path().join("organization/stock-model");
    fs::create_dir_all(&model_directory).expect("model directory should be created");
    super::write_minimal_model_config(&model_directory, "qwen3_5_moe", 262_144);
    super::write_required_model_files(&model_directory);
    fs::write(
        model_directory.join("manifest.json"),
        r#"{"name":"an export tool's own manifest"}"#,
    )
    .expect("the unrelated manifest should be written");
    assert!(
        discovered_models_under(models_root.path())
            .iter()
            .any(|discovered_model| discovered_model.model_id == "stock-model"),
        "only an Atlas manifest routes a directory to the Atlas path"
    );
}
