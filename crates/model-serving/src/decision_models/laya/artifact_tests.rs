//! Unit tests for validating a Laya artifact directory.

use std::path::Path;

use super::{LayaArtifact, LayaArtifactError, MAXIMUM_CONFIG_JSON_BYTES};

const VALID_CONFIG_JSON: &str = r#"{
    "encoder": {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "max_position_embeddings": 512, "local_attention": 8,
        "global_attn_every_n_layers": 3,
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0},
            "sliding_attention": {"rope_theta": 10000.0}
        },
        "layer_norm_eps": 1e-5, "pad_token_id": 0, "cls_token_id": 1, "sep_token_id": 2
    },
    "max_len": 64, "head_max_len": 24,
    "quantization": {"group_size": 32, "bits": 8}
}"#;

/// Writes a directory holding every file a Laya artifact needs. The weights
/// and tokenizer hold placeholder bytes: this stage only checks they exist.
fn write_artifact_directory(config_json: &str) -> tempfile::TempDir {
    let artifact_directory = tempfile::tempdir().expect("a temporary artifact directory");
    write_file(
        artifact_directory.path(),
        "config.json",
        config_json.as_bytes(),
    );
    write_file(artifact_directory.path(), "tokenizer.json", b"{}");
    write_file(artifact_directory.path(), "tokenizer_config.json", b"{}");
    write_file(artifact_directory.path(), "model.safetensors", b"weights");
    artifact_directory
}

fn write_file(artifact_directory: &Path, file_name: &str, file_bytes: &[u8]) {
    std::fs::write(artifact_directory.join(file_name), file_bytes)
        .expect("the fixture file should write");
}

fn load_error(artifact_directory: &Path) -> LayaArtifactError {
    match LayaArtifact::load(artifact_directory) {
        Ok(_) => panic!("the artifact must be refused"),
        Err(artifact_error) => artifact_error,
    }
}

#[test]
fn should_load_a_directory_holding_every_artifact_file() {
    let artifact_directory = write_artifact_directory(VALID_CONFIG_JSON);
    let artifact = LayaArtifact::load(artifact_directory.path()).expect("a complete artifact");
    assert_eq!(
        artifact
            .configuration
            .sequence_limits
            .maximum_sequence_tokens,
        64
    );
    assert_eq!(artifact.tokenizer_json_bytes, b"{}");
}

#[test]
fn should_name_each_missing_file() {
    for missing_file_name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        let artifact_directory = write_artifact_directory(VALID_CONFIG_JSON);
        std::fs::remove_file(artifact_directory.path().join(missing_file_name))
            .expect("the fixture file should delete");
        let LayaArtifactError::UnreadableFile { file_name, .. } =
            load_error(artifact_directory.path())
        else {
            panic!("a missing {missing_file_name} must be reported as unreadable");
        };
        assert_eq!(file_name, missing_file_name);
    }
}

#[test]
fn should_accept_a_config_of_exactly_the_size_limit_and_refuse_one_byte_more() {
    let padding_bytes = MAXIMUM_CONFIG_JSON_BYTES as usize - VALID_CONFIG_JSON.len();
    let config_at_the_limit = format!("{VALID_CONFIG_JSON}{}", " ".repeat(padding_bytes));
    let at_the_limit = write_artifact_directory(&config_at_the_limit);
    assert!(LayaArtifact::load(at_the_limit.path()).is_ok());
    let over_the_limit = write_artifact_directory(&format!("{config_at_the_limit} "));
    assert!(matches!(
        load_error(over_the_limit.path()),
        LayaArtifactError::FileTooLarge {
            file_name: "config.json",
            ..
        }
    ));
}

#[test]
fn should_refuse_a_config_that_is_not_json() {
    let artifact_directory = write_artifact_directory("{not json");
    assert!(matches!(
        load_error(artifact_directory.path()),
        LayaArtifactError::MalformedConfig(_)
    ));
}

#[test]
fn should_refuse_a_config_the_backend_cannot_serve() {
    let artifact_directory = write_artifact_directory(r#"{"model_type": "laya"}"#);
    assert!(matches!(
        load_error(artifact_directory.path()),
        LayaArtifactError::Configuration(_)
    ));
}
