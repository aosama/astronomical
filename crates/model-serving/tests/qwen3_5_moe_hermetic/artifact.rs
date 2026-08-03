use std::collections::BTreeMap;

use astronomical_model_serving::{
    Qwen3_5MoEArtifactError, Qwen3_5MoEMtpArtifactCapability, Qwen3_5MoEShardIndex, TensorProfile,
    qwen3_5_moe_language_tensor_profiles, qwen3_5_moe_quantized_mtp_tensor_names,
};
use serde_json::json;

use crate::common::qwen3_5_moe::{
    certified_dense_qwen3_6_config, certified_ornith_config,
    certified_qwen3_5_moe_language_tensor_profiles,
};

const CERTIFIED_LANGUAGE_PAYLOAD_BYTES: u64 = 22_164_699_392;
const CERTIFIED_TOTAL_PARAMETERS: u64 = 34_660_608_768;
const LANGUAGE_SHARD_FILE_NAMES: [&str; 5] = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
];
const VISION_SIDECAR_FILE_NAME: &str = "optiq/optiq_vision.safetensors";
const VISION_ONLY_MODEL_SHARD_FILE_NAME: &str = "model-vision-only.safetensors";

#[test]
fn should_exclude_the_fixed_vision_sidecar_from_the_executable_model_shard_inventory() {
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let index_bytes = certified_test_index_bytes();

    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the complete certified Ornith shard inventory should parse");

    assert_eq!(
        shard_index.total_payload_bytes(),
        CERTIFIED_LANGUAGE_PAYLOAD_BYTES
    );
    assert_eq!(shard_index.tensor_count(), 1_757);
    assert_eq!(shard_index.language_tensor_count(), 1_757);
    assert_eq!(
        shard_index.model_shard_file_names(),
        &LANGUAGE_SHARD_FILE_NAMES
    );
}

#[test]
fn should_include_a_vision_only_model_shard_in_the_loaded_shard_inventory() {
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let mut index_document =
        serde_json::from_slice::<serde_json::Value>(&certified_test_index_bytes())
            .expect("the certified synthetic shard index should parse");
    let weight_map = index_document
        .get_mut("weight_map")
        .and_then(serde_json::Value::as_object_mut)
        .expect("the certified synthetic shard index should contain a weight map");
    for (tensor_name, tensor_shard_file_name) in weight_map {
        if tensor_name.starts_with("vision_tower.") {
            *tensor_shard_file_name = json!(VISION_ONLY_MODEL_SHARD_FILE_NAME);
        }
    }
    let index_bytes = serde_json::to_vec(&index_document)
        .expect("the updated synthetic shard index should serialize");

    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the vision-only shard inventory should parse");

    assert!(
        shard_index
            .model_shard_file_names()
            .iter()
            .any(|shard_file_name| shard_file_name == VISION_ONLY_MODEL_SHARD_FILE_NAME)
    );
}

#[test]
fn should_reject_an_ornith_shard_index_with_an_unexpected_executable_language_tensor_name() {
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let unexpected_tensor_name = "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight";
    let index_bytes = certified_test_index_bytes_with_language_tensor_replacement(
        0,
        unexpected_tensor_name,
        &language_tensor_profiles,
    );

    assert!(matches!(
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles),
        Err(Qwen3_5MoEArtifactError::UnexpectedLanguageTensor { tensor_name })
            if tensor_name == unexpected_tensor_name
    ));
}

#[test]
fn should_classify_empty_mtp_inventory_as_target_only() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let index_bytes = certified_test_index_bytes();
    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the target-only shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MoEMtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MoEMtpArtifactCapability::TargetOnly
    );
}

#[test]
fn should_classify_complete_mtp_inventory_as_mtp_capable() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        qwen3_5_moe_quantized_mtp_tensor_names(&certified_ornith_config),
        &language_tensor_profiles,
    );
    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the MTP-capable shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MoEMtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MoEMtpArtifactCapability::MtpCapable {
            discovered_mtp_layer_count: 1,
            supported_mtp_draft_depth: 1,
            mtp_tensor_count: 42,
        }
    );
}

#[test]
fn should_classify_partial_mtp_inventory_as_invalid_mtp() {
    let certified_ornith_config = certified_ornith_config();
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    let mut mtp_tensor_names = qwen3_5_moe_quantized_mtp_tensor_names(&certified_ornith_config);
    mtp_tensor_names.remove("language_model.mtp.fc.weight");
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        mtp_tensor_names,
        &language_tensor_profiles,
    );
    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the partial MTP shard inventory should parse");

    let mtp_artifact_capability =
        Qwen3_5MoEMtpArtifactCapability::from_shard_index(&certified_ornith_config, &shard_index);

    assert!(matches!(
        mtp_artifact_capability,
        Qwen3_5MoEMtpArtifactCapability::InvalidMtp { ref reason }
            if reason.contains("missing MTP tensor language_model.mtp.fc.weight")
    ));
}

#[test]
fn should_classify_the_dense_qwen3_6_mtp_inventory_as_mtp_capable() {
    let certified_dense_qwen3_6_config = certified_dense_qwen3_6_config();
    let language_tensor_profiles =
        qwen3_5_moe_language_tensor_profiles(&certified_dense_qwen3_6_config);
    let index_bytes = certified_test_index_bytes_with_mtp_tensor_names(
        qwen3_5_moe_quantized_mtp_tensor_names(&certified_dense_qwen3_6_config),
        &language_tensor_profiles,
    );
    let shard_index =
        Qwen3_5MoEShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
            .expect("the dense Qwen3.6 MTP inventory should parse as an inventory");

    let mtp_artifact_capability = Qwen3_5MoEMtpArtifactCapability::from_shard_index(
        &certified_dense_qwen3_6_config,
        &shard_index,
    );

    assert_eq!(
        mtp_artifact_capability,
        Qwen3_5MoEMtpArtifactCapability::MtpCapable {
            discovered_mtp_layer_count: 1,
            supported_mtp_draft_depth: 1,
            mtp_tensor_count: 29,
        }
    );
}

fn certified_test_index_bytes() -> Vec<u8> {
    let language_tensor_profiles = certified_qwen3_5_moe_language_tensor_profiles();
    certified_test_index_bytes_with_optional_language_tensor_replacement(
        None,
        &language_tensor_profiles,
    )
}

fn certified_test_index_bytes_with_mtp_tensor_names(
    mtp_tensor_names: impl IntoIterator<Item = String>,
    language_tensor_profiles: &[TensorProfile],
) -> Vec<u8> {
    certified_test_index_bytes_with_optional_language_tensor_and_mtp_tensor_replacement(
        None,
        mtp_tensor_names,
        language_tensor_profiles,
    )
}

fn certified_test_index_bytes_with_language_tensor_replacement(
    replacement_tensor_index: usize,
    replacement_tensor_name: &str,
    language_tensor_profiles: &[TensorProfile],
) -> Vec<u8> {
    certified_test_index_bytes_with_optional_language_tensor_replacement(
        Some((replacement_tensor_index, replacement_tensor_name)),
        language_tensor_profiles,
    )
}

fn certified_test_index_bytes_with_optional_language_tensor_replacement(
    replacement_tensor: Option<(usize, &str)>,
    language_tensor_profiles: &[TensorProfile],
) -> Vec<u8> {
    certified_test_index_bytes_with_optional_language_tensor_and_mtp_tensor_replacement(
        replacement_tensor,
        std::iter::empty(),
        language_tensor_profiles,
    )
}

fn certified_test_index_bytes_with_optional_language_tensor_and_mtp_tensor_replacement(
    replacement_tensor: Option<(usize, &str)>,
    mtp_tensor_names: impl IntoIterator<Item = String>,
    language_tensor_profiles: &[TensorProfile],
) -> Vec<u8> {
    let mut weight_map = BTreeMap::new();
    for (language_tensor_index, tensor_profile) in language_tensor_profiles.iter().enumerate() {
        let tensor_name = replacement_tensor
            .filter(|(replacement_tensor_index, _)| {
                *replacement_tensor_index == language_tensor_index
            })
            .map_or(
                tensor_profile.name.as_str(),
                |(_, replacement_tensor_name)| replacement_tensor_name,
            );
        weight_map.insert(
            tensor_name.to_owned(),
            LANGUAGE_SHARD_FILE_NAMES[language_tensor_index % LANGUAGE_SHARD_FILE_NAMES.len()],
        );
    }
    for mtp_tensor_name in mtp_tensor_names {
        weight_map.insert(mtp_tensor_name, LANGUAGE_SHARD_FILE_NAMES[0]);
    }
    for vision_tensor_index in 0..333 {
        weight_map.insert(
            format!("vision_tower.synthetic.{vision_tensor_index}.weight"),
            VISION_SIDECAR_FILE_NAME,
        );
    }
    serde_json::to_vec(&json!({
        "metadata": {
            "total_size": CERTIFIED_LANGUAGE_PAYLOAD_BYTES,
            "total_parameters": CERTIFIED_TOTAL_PARAMETERS,
        },
        "weight_map": weight_map,
    }))
    .expect("the bounded synthetic shard index should serialize")
}
