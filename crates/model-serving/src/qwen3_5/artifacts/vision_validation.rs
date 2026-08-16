use std::collections::BTreeSet;

use super::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use super::{Qwen3_5ArtifactError, Qwen3_5ShardIndex, Qwen3_5VisionConfig};

/// Physical storage validated for a Qwen3.5 visual tower.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValidatedVisionTowerStorage {
    Absent,
    EmbeddedInModelShards,
    SeparateSidecar,
}

impl ValidatedVisionTowerStorage {
    #[must_use]
    pub(super) const fn has_validated_vision_tower(self) -> bool {
        !matches!(self, Self::Absent)
    }

    #[must_use]
    pub(super) const fn has_separate_sidecar(self) -> bool {
        matches!(self, Self::SeparateSidecar)
    }
}

/// Verifies that the shard index has either no visual tower or one complete tower.
pub(super) fn validate_vision_tower_inventory(
    shard_index: &Qwen3_5ShardIndex,
    vision_config: Option<&Qwen3_5VisionConfig>,
) -> Result<ValidatedVisionTowerStorage, Qwen3_5ArtifactError> {
    let vision_tensor_name_to_shard_file_name = shard_index.vision_tensor_name_to_shard_file_name();
    if vision_tensor_name_to_shard_file_name.is_empty() {
        return Ok(ValidatedVisionTowerStorage::Absent);
    }
    let vision_config = vision_config.ok_or(Qwen3_5ArtifactError::MissingVisionConfig)?;

    let expected_vision_tensor_names = qwen3_5_vision_tensor_profiles(vision_config)
        .into_iter()
        .map(|tensor_profile| tensor_profile.name)
        .collect::<BTreeSet<_>>();
    let actual_vision_tensor_names = vision_tensor_name_to_shard_file_name
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if let Some(unexpected_tensor_name) = actual_vision_tensor_names
        .difference(&expected_vision_tensor_names)
        .next()
    {
        return Err(Qwen3_5ArtifactError::UnexpectedVisionTensor {
            tensor_name: unexpected_tensor_name.clone(),
        });
    }
    if let Some(missing_tensor_name) = expected_vision_tensor_names
        .difference(&actual_vision_tensor_names)
        .next()
    {
        return Err(Qwen3_5ArtifactError::MissingVisionTensor {
            tensor_name: missing_tensor_name.clone(),
        });
    }

    let uses_separate_sidecar = vision_tensor_name_to_shard_file_name
        .values()
        .any(|shard_file_name| shard_index.is_vision_sidecar_file(shard_file_name));
    if uses_separate_sidecar {
        if let Some((tensor_name, shard_file_name)) = vision_tensor_name_to_shard_file_name
            .iter()
            .find(|(_, shard_file_name)| !shard_index.is_vision_sidecar_file(shard_file_name))
        {
            return Err(Qwen3_5ArtifactError::MixedVisionTensorStorage {
                tensor_name: tensor_name.clone(),
                shard_file_name: shard_file_name.clone(),
            });
        }
        return Ok(ValidatedVisionTowerStorage::SeparateSidecar);
    }

    for (tensor_name, shard_file_name) in vision_tensor_name_to_shard_file_name {
        if !shard_index
            .model_shard_file_names()
            .iter()
            .any(|language_shard_file_name| language_shard_file_name == shard_file_name)
        {
            return Err(Qwen3_5ArtifactError::VisionTensorOutsideModelShards {
                tensor_name: tensor_name.clone(),
                shard_file_name: shard_file_name.clone(),
            });
        }
    }
    Ok(ValidatedVisionTowerStorage::EmbeddedInModelShards)
}
