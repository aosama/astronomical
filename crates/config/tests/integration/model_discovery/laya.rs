//! Shallow discovery contracts for executable Laya decision-model artifacts,
//! including the checkpoints one repository bundles in subfolders.

use std::fs;
use std::path::Path;

use astronomical_config::{DecisionModelCapabilities, DiscoveredModel, ModelCapabilities};

/// Writes the minimal Laya checkpoint accepted by shallow discovery.
fn write_laya_checkpoint(checkpoint_directory: &Path, laya_config_json: &serde_json::Value) {
    fs::create_dir_all(checkpoint_directory).expect("checkpoint directory should be created");
    fs::write(
        checkpoint_directory.join("config.json"),
        laya_config_json.to_string(),
    )
    .expect("laya config should be written");
    fs::write(
        checkpoint_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("tokenizer should be written");
    fs::write(checkpoint_directory.join("tokenizer_config.json"), "{}")
        .expect("tokenizer config should be written");
    fs::write(
        checkpoint_directory.join("model.safetensors"),
        b"fictional-weights",
    )
    .expect("safetensors should be written");
}

fn laya_config_json(maximum_tokens_per_question: Option<u32>) -> serde_json::Value {
    let mut laya_config_json = serde_json::json!({
        "model_type": "laya",
        "encoder": { "model_type": "modernbert", "hidden_size": 1_024 },
        "quantization": { "group_size": 32, "bits": 8 },
    });
    if let Some(maximum_tokens_per_question) = maximum_tokens_per_question {
        laya_config_json["max_len"] = serde_json::json!(maximum_tokens_per_question);
    }
    laya_config_json
}

fn discovered_models_in(temporary_directory: &tempfile::TempDir) -> Vec<DiscoveredModel> {
    super::discover_configured_models(temporary_directory)
        .into_iter()
        .flat_map(|directory_scan| directory_scan.discovered_models)
        .collect()
}

fn maximum_tokens_per_question(discovered_model: &DiscoveredModel) -> u32 {
    match &discovered_model.capabilities {
        ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question,
        }) => *maximum_tokens_per_question,
        other_capabilities => panic!("expected decisions capability, got {other_capabilities:?}"),
    }
}

#[test]
fn should_discover_a_laya_decision_model_with_its_stated_token_limit() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_laya_checkpoint(
        &temporary_directory.path().join("laya-MLX-8bit"),
        &laya_config_json(Some(1_024)),
    );

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models.len(), 1, "Laya must be discovered");
    assert_eq!(discovered_models[0].model_id, "laya-MLX-8bit");
    assert_eq!(
        discovered_models[0].model_family,
        astronomical_config::ModelFamily::Laya
    );
    assert_eq!(maximum_tokens_per_question(&discovered_models[0]), 1_024);
}

#[test]
fn should_default_the_token_limit_when_the_config_states_none() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_laya_checkpoint(
        &temporary_directory.path().join("laya-MLX-8bit"),
        &laya_config_json(None),
    );

    let discovered_models = discovered_models_in(&temporary_directory);

    assert_eq!(discovered_models.len(), 1);
    assert_eq!(maximum_tokens_per_question(&discovered_models[0]), 512);
}

#[test]
fn should_discover_each_bundled_checkpoint_as_its_own_model() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let repository_directory = temporary_directory.path().join("laya-MLX-8bit");
    write_laya_checkpoint(&repository_directory, &laya_config_json(Some(512)));
    write_laya_checkpoint(
        &repository_directory.join("multilingual"),
        &laya_config_json(Some(1_024)),
    );

    let mut discovered_models = discovered_models_in(&temporary_directory);
    discovered_models.sort_by(|left, right| left.model_id.cmp(&right.model_id));

    let discovered_model_ids: Vec<&str> = discovered_models
        .iter()
        .map(|discovered_model| discovered_model.model_id.as_str())
        .collect();
    assert_eq!(
        discovered_model_ids,
        ["laya-MLX-8bit", "laya-MLX-8bit-multilingual"]
    );
    assert_eq!(maximum_tokens_per_question(&discovered_models[0]), 512);
    assert_eq!(maximum_tokens_per_question(&discovered_models[1]), 1_024);
    assert_eq!(
        discovered_models[1].model_directory,
        repository_directory.join("multilingual"),
        "a bundled checkpoint loads from its own subfolder"
    );
}

#[test]
fn should_discover_bundled_checkpoints_inside_a_hugging_face_cache_snapshot() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let snapshot_directory = temporary_directory
        .path()
        .join("models--example-org--laya-MLX-8bit")
        .join("snapshots")
        .join("0123456789abcdef0123456789abcdef01234567");
    write_laya_checkpoint(&snapshot_directory, &laya_config_json(Some(512)));
    write_laya_checkpoint(
        &snapshot_directory.join("multilingual"),
        &laya_config_json(Some(1_024)),
    );

    let mut discovered_model_ids: Vec<String> = discovered_models_in(&temporary_directory)
        .into_iter()
        .map(|discovered_model| discovered_model.model_id)
        .collect();
    discovered_model_ids.sort();

    assert_eq!(
        discovered_model_ids,
        ["laya-MLX-8bit", "laya-MLX-8bit-multilingual"]
    );
}

#[test]
fn should_skip_subfolders_that_are_not_complete_laya_checkpoints() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let repository_directory = temporary_directory.path().join("laya-MLX-8bit");
    write_laya_checkpoint(&repository_directory, &laya_config_json(Some(512)));
    // A Laya config with no weights beside it, a folder of something else,
    // and a hidden folder: none is a checkpoint that can be served.
    let weightless_directory = repository_directory.join("weightless");
    fs::create_dir(&weightless_directory).expect("weightless directory should be created");
    fs::write(
        weightless_directory.join("config.json"),
        laya_config_json(Some(512)).to_string(),
    )
    .expect("weightless config should be written");
    let unrelated_directory = repository_directory.join("assets");
    fs::create_dir(&unrelated_directory).expect("assets directory should be created");
    fs::write(
        unrelated_directory.join("config.json"),
        r#"{"model_type":"other"}"#,
    )
    .expect("unrelated config should be written");
    write_laya_checkpoint(
        &repository_directory.join(".hidden"),
        &laya_config_json(Some(512)),
    );

    let discovered_model_ids: Vec<String> = discovered_models_in(&temporary_directory)
        .into_iter()
        .map(|discovered_model| discovered_model.model_id)
        .collect();

    assert_eq!(discovered_model_ids, ["laya-MLX-8bit"]);
}

#[test]
fn should_reject_a_laya_model_without_its_tokenizer_config() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("laya-incomplete");
    write_laya_checkpoint(&model_directory, &laya_config_json(Some(512)));
    fs::remove_file(model_directory.join("tokenizer_config.json"))
        .expect("tokenizer config should be removable");

    assert!(
        discovered_models_in(&temporary_directory).is_empty(),
        "incomplete Laya artifacts must stay undiscoverable"
    );
}

#[test]
fn should_reject_unsupported_laya_quantization_widths() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let mut four_bit_config_json = laya_config_json(Some(512));
    four_bit_config_json["quantization"]["bits"] = serde_json::json!(4);
    write_laya_checkpoint(
        &temporary_directory.path().join("laya-4bit"),
        &four_bit_config_json,
    );

    assert!(
        discovered_models_in(&temporary_directory).is_empty(),
        "only the reviewed 8-bit profile is executable"
    );
}

#[test]
fn should_reject_a_laya_config_without_its_encoder_section() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let mut encoderless_config_json = laya_config_json(Some(512));
    encoderless_config_json
        .as_object_mut()
        .expect("the config should be an object")
        .remove("encoder");
    write_laya_checkpoint(
        &temporary_directory.path().join("laya-no-encoder"),
        &encoderless_config_json,
    );

    assert!(discovered_models_in(&temporary_directory).is_empty());
}

#[test]
fn should_reject_a_laya_config_with_a_zero_token_limit() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_laya_checkpoint(
        &temporary_directory.path().join("laya-zero-limit"),
        &laya_config_json(Some(0)),
    );

    assert!(discovered_models_in(&temporary_directory).is_empty());
}
