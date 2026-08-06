use astronomical_model_serving::{
    Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, qwen3_5_mtp_tensor_names,
};
use serde_json::Value;

use crate::common::qwen3_5_moe::{
    certified_ornith_config, certified_qwen3_5_language_tensor_profiles,
};
use crate::qwen3_5_hermetic::artifact_test_support::{
    certified_test_index_bytes, certified_test_index_bytes_with_mtp_tensor_names,
};

#[test]
fn should_classify_empty_mtp_inventory_as_target_only() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    let index_bytes = certified_test_index_bytes();
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the target-only shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MtpArtifactCapability::TargetOnly
    );
}

#[test]
fn should_classify_complete_mtp_inventory_as_mtp_capable() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        qwen3_5_mtp_tensor_names(&certified_ornith_config),
        &language_tensor_profiles,
    );
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the MTP-capable shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MtpArtifactCapability::MtpCapable {
            discovered_mtp_layer_count: 1,
            supported_mtp_draft_depth: 1,
            mtp_tensor_count: 42,
        }
    );
}

#[test]
fn should_fall_back_to_target_only_for_a_partial_mtp_inventory() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    let mut mtp_tensor_names = qwen3_5_mtp_tensor_names(&certified_ornith_config);
    mtp_tensor_names.remove("language_model.mtp.fc.weight");
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        mtp_tensor_names,
        &language_tensor_profiles,
    );
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the partial MTP shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MtpArtifactCapability::TargetOnly
    );
}

#[test]
fn should_omit_an_absent_mtp_only_shard_without_removing_target_ownership() {
    let ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    let mut index_document =
        serde_json::from_slice::<Value>(&certified_test_index_bytes_with_mtp_tensor_names(
            qwen3_5_mtp_tensor_names(&ornith_config),
            &language_tensor_profiles,
        ))
        .expect("the synthetic MTP index should decode");
    let weight_map = index_document["weight_map"]
        .as_object_mut()
        .expect("the synthetic index should contain a weight map");
    for (tensor_name, shard_file_name) in weight_map {
        if tensor_name.starts_with("language_model.mtp.") {
            *shard_file_name = Value::String("predictor-weights.safetensors".to_owned());
        }
    }
    let index_bytes =
        serde_json::to_vec(&index_document).expect("the synthetic MTP-only index should serialize");
    let mut shard_index =
        Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the synthetic MTP-only index should parse");

    assert!(shard_index.is_mtp_only_shard_file("predictor-weights.safetensors"));
    assert!(
        shard_index
            .model_shard_file_names()
            .iter()
            .all(|shard_file_name| shard_file_name != "predictor-weights.safetensors")
    );
    assert!(shard_index.omit_optional_mtp_shard_file("predictor-weights.safetensors"));
    assert_eq!(shard_index.mtp_tensor_count(), 0);
    assert!(
        shard_index
            .model_shard_file_names()
            .iter()
            .all(|shard_file_name| shard_file_name != "predictor-weights.safetensors")
    );
    assert_eq!(
        shard_index.language_tensor_count(),
        language_tensor_profiles.len()
    );
}
