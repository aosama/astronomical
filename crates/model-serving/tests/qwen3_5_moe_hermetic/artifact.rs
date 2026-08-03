use astronomical_model_serving::{
    Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, qwen3_5_quantized_mtp_tensor_names,
};

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
        qwen3_5_quantized_mtp_tensor_names(&certified_ornith_config),
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
fn should_classify_partial_mtp_inventory_as_invalid_mtp() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    let mut mtp_tensor_names = qwen3_5_quantized_mtp_tensor_names(&certified_ornith_config);
    mtp_tensor_names.remove("language_model.mtp.fc.weight");
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        mtp_tensor_names,
        &language_tensor_profiles,
    );
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the partial MTP shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert!(matches!(
        mtp_artifact_capability,
        Qwen3_5MtpArtifactCapability::InvalidMtp { ref reason }
            if reason.contains("missing MTP tensor language_model.mtp.fc.weight")
    ));
}
