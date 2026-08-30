use astronomical_model_serving::{
    Qwen3_5MtpArtifactCapability, Qwen3_5ShardIndex, qwen3_5_language_tensor_profiles,
    qwen3_5_mtp_tensor_names,
};

use crate::common::qwen3_5::frozen_dense_qwen3_6_config;
use crate::qwen3_5_hermetic::artifact_test_support::frozen_test_index_bytes_with_mtp_tensor_names;

#[test]
fn should_classify_the_dense_qwen3_6_mtp_inventory_as_mtp_capable() {
    let frozen_dense_qwen3_6_config = frozen_dense_qwen3_6_config();
    let language_tensor_profiles = qwen3_5_language_tensor_profiles(&frozen_dense_qwen3_6_config);
    let index_bytes = frozen_test_index_bytes_with_mtp_tensor_names(
        qwen3_5_mtp_tensor_names(&frozen_dense_qwen3_6_config),
        &language_tensor_profiles,
    );
    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the dense Qwen3.6 MTP inventory should parse as an inventory");

    let mtp_artifact_capability =
        Qwen3_5MtpArtifactCapability::from_shard_index(&frozen_dense_qwen3_6_config, &shard_index);

    assert!(
        matches!(
            mtp_artifact_capability,
            Qwen3_5MtpArtifactCapability::MtpCapable {
                stored_mtp_layer_count: 1,
                mtp_tensor_count,
                ..
            } if mtp_tensor_count > 0
        ),
        "dense MTP inventory should be capable: {mtp_artifact_capability:?}"
    );
}
