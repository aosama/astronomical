use astronomical_model_serving::{
    Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, qwen3_5_language_tensor_profiles,
    qwen3_5_quantized_mtp_tensor_names,
};

use crate::common::qwen3_5::certified_dense_qwen3_6_config;
use crate::qwen3_5_hermetic::artifact_test_support::certified_test_index_bytes_with_mtp_tensor_names;

#[test]
fn should_classify_the_dense_qwen3_6_mtp_inventory_as_mtp_capable() {
    let certified_dense_qwen3_6_config = certified_dense_qwen3_6_config();
    let language_tensor_profiles =
        qwen3_5_language_tensor_profiles(&certified_dense_qwen3_6_config);
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        qwen3_5_quantized_mtp_tensor_names(&certified_dense_qwen3_6_config),
        &language_tensor_profiles,
    );
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the dense Qwen3.6 MTP inventory should parse as an inventory");

    let mtp_artifact_capability = Qwen3_5MtpArtifactCapability::from_shard_index(
        &certified_dense_qwen3_6_config,
        &shard_index,
    );

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MtpArtifactCapability::MtpCapable {
            discovered_mtp_layer_count: 1,
            supported_mtp_draft_depth: 1,
            mtp_tensor_count: 29,
        }
    );
}
