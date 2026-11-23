//! Complete shallow Qwen-Image-2.1 fixture used to prove the worker's selected-directory trust
//! boundary. The documents mirror what `verify_qwen_image_21_model_directory` accepts, with
//! placeholder weights whose bytes never matter to the factory's CPU-side checks.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use test_harness::fixtures::image_pipeline_fixture::{
    qwen_image_21_pipeline_index, qwen_image_21_scheduler_config,
    qwen_image_21_text_encoder_config, qwen_image_21_transformer_config, qwen_image_21_vae_config,
    write_component_weights, write_json,
};

pub(super) const CANONICAL_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
pub(super) const REVIEWED_REVISION: &str = "4db4e8c0c0e7a1debf0320415bec8388e888494c";

pub(super) fn write_executable_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "processor",
        "scheduler",
        "text_encoder",
        "transformer",
        "vae",
    ] {
        fs::create_dir_all(model_directory.join(relative_directory))
            .expect("Qwen fixture directory should be created");
    }
    write_json(
        &model_directory.join("model_index.json"),
        qwen_image_21_pipeline_index(),
    );
    write_json(
        &model_directory.join("transformer/config.json"),
        qwen_image_21_transformer_config(),
    );
    write_json(
        &model_directory.join("text_encoder/config.json"),
        qwen_image_21_text_encoder_config(),
    );
    write_json(
        &model_directory.join("vae/config.json"),
        qwen_image_21_vae_config(),
    );
    write_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        qwen_image_21_scheduler_config(),
    );
    for processor_file in [
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ] {
        fs::write(
            model_directory.join("processor").join(processor_file),
            "fixture",
        )
        .expect("processor fixture should be written");
    }
    write_component_weights(&model_directory.join("text_encoder"), 11);
    write_component_weights(&model_directory.join("transformer"), 17);
    write_component_weights(&model_directory.join("vae"), 23);
    fs::write(
        model_directory.join("README.md"),
        "---\nlicense: other\nlicense_name: qwen-research\nbase_model: Qwen/Qwen-Image-2.1\n---\n# Qwen-Image-2.1\n",
    )
    .expect("Qwen license provenance should be written");
    write_revision(model_directory, REVIEWED_REVISION);
}

pub(super) fn write_revision(model_directory: &Path, revision: &str) {
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{revision}\nfixture-etag\n0\n"),
    )
    .expect("revision metadata should be written");
}

/// Mutates one transformer field so exact-directory verification no longer accepts the artifact
/// while the revision provenance stays intact.
pub(super) fn mutate_profile(model_directory: &Path) {
    let config_path = model_directory.join("transformer/config.json");
    let config_bytes = fs::read(&config_path).expect("transformer config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("transformer config should parse");
    config_document["num_layers"] = json!(31);
    write_json(&config_path, config_document);
}

/// Removes the VAE weights so the component inventory no longer matches the reviewed profile.
pub(super) fn remove_component(model_directory: &Path) {
    fs::remove_file(model_directory.join("vae/model.safetensors"))
        .expect("the VAE weights should be removed");
}
