use std::fs;
use std::path::Path;

use crate::test_fixtures::{FixtureTensor, open_directory, seeded_values, write_bfloat16_file};

use super::*;

const TEXT_TENSOR: &str = "language_model.model.layers.0.norm.weight";

fn write_text_only_weights(directory: &Path) {
    write_bfloat16_file(
        &directory.join("model.safetensors"),
        &[FixtureTensor {
            name: TEXT_TENSOR,
            float_values: seeded_values(1, 4),
        }],
    );
}

fn write_tower_sidecar(directory: &Path, block_count: u32) {
    let mut block_names = Vec::new();
    for block_index in 0..block_count {
        block_names.push(block_tensor_name(block_index));
    }
    let mut tensors = vec![FixtureTensor {
        name: "vision_tower.merger.norm.weight",
        float_values: seeded_values(2, 6),
    }];
    for block_name in &block_names {
        tensors.push(FixtureTensor {
            name: block_name,
            float_values: seeded_values(3, 2),
        });
    }
    write_bfloat16_file(&directory.join("optiq/optiq_vision.safetensors"), &tensors);
}

fn block_tensor_name(block_index: u32) -> String {
    format!("vision_tower.blocks.{block_index}.attn.proj.weight")
}

async fn inspect_directory(directory: &Path) -> TowerInspection {
    inspect_artifact(&open_directory(directory).await)
        .await
        .unwrap()
}

#[tokio::test]
async fn should_find_a_complete_tower_in_a_sidecar_the_index_does_not_name() {
    let directory = tempfile::tempdir().unwrap();
    write_text_only_weights(directory.path());
    write_tower_sidecar(directory.path(), 2);
    fs::write(
        directory.path().join("config.json"),
        r#"{"image_token_id": 9, "text_config": {"hidden_size": 8},
            "vision_config": {"depth": 2, "out_hidden_size": 8}}"#,
    )
    .unwrap();
    fs::write(
        directory.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map": {{"{TEXT_TENSOR}": "model.safetensors"}}}}"#),
    )
    .unwrap();

    let inspection = inspect_directory(directory.path()).await;

    assert_eq!(inspection.status, TowerStatus::Present);
    assert_eq!(inspection.weight_file_count, 2);
    assert_eq!(inspection.total_tensor_count, 4);
    let tower = inspection.tower.unwrap();
    assert_eq!(tower.tensor_count, 3);
    assert_eq!(tower.parameter_count, 10);
    assert_eq!(tower.byte_count, 20);
    assert_eq!(tower.block_count, 2);
    assert_eq!(tower.name_prefixes, ["vision_tower"]);
    assert_eq!(tower.weight_files, ["optiq/optiq_vision.safetensors"]);
    assert_eq!(tower.data_type_counts["BF16"], 3);
    assert!(!tower.is_quantized);
    assert_eq!(tower.is_in_weight_index, Some(false));
}

#[tokio::test]
async fn should_report_a_tower_that_disagrees_with_the_config_as_partial() {
    let directory = tempfile::tempdir().unwrap();
    write_tower_sidecar(directory.path(), 2);
    fs::write(
        directory.path().join("config.json"),
        r#"{"vision_config": {"depth": 27}}"#,
    )
    .unwrap();
    assert_eq!(
        inspect_directory(directory.path()).await.status,
        TowerStatus::Partial
    );
}

#[tokio::test]
async fn should_report_a_tower_with_no_vision_config_as_partial() {
    let directory = tempfile::tempdir().unwrap();
    write_tower_sidecar(directory.path(), 1);
    assert_eq!(
        inspect_directory(directory.path()).await.status,
        TowerStatus::Partial
    );
}

#[tokio::test]
async fn should_tell_image_tokens_apart_from_a_plain_text_model() {
    let directory = tempfile::tempdir().unwrap();
    write_text_only_weights(directory.path());
    fs::write(
        directory.path().join("config.json"),
        r#"{"image_token_id": 9}"#,
    )
    .unwrap();
    let inspection = inspect_directory(directory.path()).await;
    assert_eq!(inspection.status, TowerStatus::AbsentWithImageTokens);
    assert_eq!(inspection.tower, None);

    fs::write(
        directory.path().join("config.json"),
        r#"{"model_type": "text"}"#,
    )
    .unwrap();
    assert_eq!(
        inspect_directory(directory.path()).await.status,
        TowerStatus::Absent
    );
}

#[tokio::test]
async fn should_report_a_vision_config_with_no_tensors_behind_it() {
    let directory = tempfile::tempdir().unwrap();
    write_text_only_weights(directory.path());
    fs::write(
        directory.path().join("config.json"),
        r#"{"vision_config": {"depth": 2}}"#,
    )
    .unwrap();
    assert_eq!(
        inspect_directory(directory.path()).await.status,
        TowerStatus::AbsentWithVisionConfig
    );
}

#[tokio::test]
async fn should_answer_for_a_directory_with_no_weights_or_config() {
    let directory = tempfile::tempdir().unwrap();
    let inspection = inspect_directory(directory.path()).await;
    assert_eq!(inspection.status, TowerStatus::Absent);
    assert_eq!(inspection.total_tensor_count, 0);
}

#[tokio::test]
async fn should_flag_a_quantized_tower() {
    let directory = tempfile::tempdir().unwrap();
    write_bfloat16_file(
        &directory.path().join("model.safetensors"),
        &[FixtureTensor {
            name: "vision_tower.blocks.0.attn.proj.scales",
            float_values: seeded_values(4, 2),
        }],
    );
    let inspection = inspect_directory(directory.path()).await;
    assert!(inspection.tower.unwrap().is_quantized);
}

#[tokio::test]
async fn should_fail_on_a_config_that_is_not_json() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("config.json"), "{ not json").unwrap();
    let source = open_directory(directory.path()).await;
    assert!(matches!(
        inspect_artifact(&source).await,
        Err(VisionTowerError::JsonMalformed { .. })
    ));
}

#[tokio::test]
async fn should_fail_on_a_truncated_weight_file() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("model.safetensors"), [0_u8; 3]).unwrap();
    let source = open_directory(directory.path()).await;
    assert!(inspect_artifact(&source).await.is_err());
}
