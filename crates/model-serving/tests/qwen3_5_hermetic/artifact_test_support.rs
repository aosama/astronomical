use std::collections::BTreeMap;

use astronomical_model_serving::TensorProfile;
use serde_json::json;

use crate::common::qwen3_5_moe::certified_qwen3_5_language_tensor_profiles;

pub(crate) const CERTIFIED_LANGUAGE_PAYLOAD_BYTES: u64 = 22_164_699_392;
const CERTIFIED_TOTAL_PARAMETERS: u64 = 34_660_608_768;
pub(crate) const LANGUAGE_SHARD_FILE_NAMES: [&str; 5] = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
];
const VISION_SIDECAR_FILE_NAME: &str = "vision/weights.safetensors";

pub(crate) fn certified_test_index_bytes() -> Vec<u8> {
    let language_tensor_profiles = certified_qwen3_5_language_tensor_profiles();
    certified_test_index_bytes_with_optional_language_tensor_replacement(
        None,
        &language_tensor_profiles,
    )
}

pub(crate) fn certified_test_index_bytes_with_mtp_tensor_names(
    mtp_tensor_names: impl IntoIterator<Item = String>,
    language_tensor_profiles: &[TensorProfile],
) -> Vec<u8> {
    certified_test_index_bytes_with_optional_language_tensor_and_mtp_tensor_replacement(
        None,
        mtp_tensor_names,
        language_tensor_profiles,
    )
}

pub(crate) fn certified_test_index_bytes_with_language_tensor_replacement(
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
