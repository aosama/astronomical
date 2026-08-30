use astronomical_model_serving::{Qwen3_5ArtifactError, Qwen3_5ShardIndex};
use serde_json::json;

use super::artifact_test_support::{
    FROZEN_LANGUAGE_PAYLOAD_BYTES, LANGUAGE_SHARD_FILE_NAMES, frozen_test_index_bytes,
    frozen_test_index_bytes_with_language_tensor_replacement,
};
use crate::common::qwen3_5_moe::expected_qwen3_5_language_tensor_profiles;

const VISION_ONLY_MODEL_SHARD_FILE_NAME: &str = "model-vision-only.safetensors";

#[test]
fn should_exclude_a_nested_vision_sidecar_from_the_executable_model_shard_inventory() {
    let language_tensor_profiles = expected_qwen3_5_language_tensor_profiles();
    let index_bytes = frozen_test_index_bytes();

    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the complete frozen Ornith 1.0 shard inventory should parse");

    assert_eq!(
        shard_index.total_payload_bytes(),
        FROZEN_LANGUAGE_PAYLOAD_BYTES
    );
    assert_eq!(shard_index.tensor_count(), 1_757);
    assert_eq!(shard_index.language_tensor_count(), 1_757);
    assert_eq!(
        shard_index.model_shard_file_names(),
        &LANGUAGE_SHARD_FILE_NAMES
    );
}

#[test]
fn should_classify_a_root_vision_only_file_by_tensor_role_instead_of_filename() {
    let language_tensor_profiles = expected_qwen3_5_language_tensor_profiles();
    let mut index_document =
        serde_json::from_slice::<serde_json::Value>(&frozen_test_index_bytes())
            .expect("the frozen synthetic shard index should parse");
    let weight_map = index_document
        .get_mut("weight_map")
        .and_then(serde_json::Value::as_object_mut)
        .expect("the frozen synthetic shard index should contain a weight map");
    for (tensor_name, tensor_shard_file_name) in weight_map {
        if tensor_name.starts_with("vision_tower.") {
            *tensor_shard_file_name = json!(VISION_ONLY_MODEL_SHARD_FILE_NAME);
        }
    }
    let index_bytes = serde_json::to_vec(&index_document)
        .expect("the updated synthetic shard index should serialize");

    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the vision-only shard inventory should parse");

    assert!(
        !shard_index
            .model_shard_file_names()
            .iter()
            .any(|shard_file_name| shard_file_name == VISION_ONLY_MODEL_SHARD_FILE_NAME)
    );
    assert_eq!(
        shard_index.vision_sidecar_file_names(),
        &[VISION_ONLY_MODEL_SHARD_FILE_NAME]
    );
}

#[test]
fn should_reject_an_ornith_shard_index_with_an_unexpected_executable_language_tensor_name() {
    let language_tensor_profiles = expected_qwen3_5_language_tensor_profiles();
    let unexpected_tensor_name = "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight";
    let index_bytes = frozen_test_index_bytes_with_language_tensor_replacement(
        0,
        unexpected_tensor_name,
        &language_tensor_profiles,
    );

    assert!(matches!(
        Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles),
        Err(Qwen3_5ArtifactError::UnexpectedLanguageTensor { tensor_name })
            if tensor_name == unexpected_tensor_name
    ));
}
