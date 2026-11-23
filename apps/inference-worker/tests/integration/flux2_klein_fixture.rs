//! Complete shallow FLUX fixture used to prove the worker's selected-directory trust boundary.

use std::fs;
use std::path::Path;

use serde_json::Value;
use test_harness::fixtures::image_pipeline_fixture::{
    write_flux2_klein_pipeline_configs, write_json, write_safetensors_payload,
};

pub(super) const CANONICAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub(super) const REVIEWED_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";

pub(super) fn write_executable_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ] {
        fs::create_dir_all(model_directory.join(relative_directory))
            .expect("FLUX fixture directory should be created");
    }
    write_flux2_klein_pipeline_configs(model_directory);
    fs::write(
        model_directory.join("text_encoder/model.safetensors.index.json"),
        r#"{"metadata":{"total_size":24},"weight_map":{"first":"model-00001-of-00002.safetensors","second":"model-00002-of-00002.safetensors"}}"#,
    )
    .expect("text encoder index should be written");
    for (relative_path, size_bytes) in [
        ("text_encoder/model-00001-of-00002.safetensors", 11),
        ("text_encoder/model-00002-of-00002.safetensors", 13),
    ] {
        write_safetensors_payload(&model_directory.join(relative_path), size_bytes);
    }
    for (relative_path, size_bytes) in [
        ("transformer/diffusion_pytorch_model.safetensors", 17),
        ("vae/diffusion_pytorch_model.safetensors", 13),
    ] {
        fs::write(model_directory.join(relative_path), vec![0_u8; size_bytes])
            .expect("modular weight should be written");
    }
    for relative_path in [
        "text_encoder/generation_config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
    ] {
        fs::write(model_directory.join(relative_path), "{}")
            .expect("modular sidecar should be written");
    }
    fs::write(
        model_directory.join("LICENSE.md"),
        "Apache License\nVersion 2.0, January 2004\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\nEND OF TERMS AND CONDITIONS\n",
    )
    .expect("Apache-2.0 license should be written");
    write_revision(model_directory, REVIEWED_REVISION);
}

pub(super) fn write_revision(model_directory: &Path, revision: &str) {
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{revision}\nfixture-etag\n0\n"),
    )
    .expect("revision metadata should be written");
}

pub(super) fn replace_json_field(config_path: &Path, field_name: &str, replacement: Value) {
    let config_bytes = fs::read(config_path).expect("component config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("component config should parse");
    config_document[field_name] = replacement;
    write_json(config_path, config_document);
}
