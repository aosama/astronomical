use std::collections::{BTreeSet, HashMap, HashSet};

use crate::ArtifactValidationError;
use crate::validated_artifact::ValidatedRequiredFile;

use super::vision_tensor_spec::qwen3_5_vision_tensor_profiles;
use super::{Qwen3_5ArtifactError, Qwen3_5ShardIndex, Qwen3_5VisionConfig};

pub(super) const VISION_SIDECAR_FILE_NAME: &str = "optiq/optiq_vision.safetensors";

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
        .any(|shard_file_name| shard_file_name == VISION_SIDECAR_FILE_NAME);
    if uses_separate_sidecar {
        if let Some((tensor_name, shard_file_name)) = vision_tensor_name_to_shard_file_name
            .iter()
            .find(|(_, shard_file_name)| *shard_file_name != VISION_SIDECAR_FILE_NAME)
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

/// Returns strict visual tensor profiles stored in one embedded model shard.
#[must_use]
pub(super) fn embedded_vision_tensor_profiles_for_shard(
    vision_tensor_profiles: &[crate::TensorProfile],
    shard_index: &Qwen3_5ShardIndex,
    shard_file_name: &str,
) -> Vec<crate::TensorProfile> {
    vision_tensor_profiles
        .iter()
        .filter(|tensor_profile| {
            shard_index
                .vision_tensor_name_to_shard_file_name()
                .get(&tensor_profile.name)
                .is_some_and(|vision_shard_file_name| vision_shard_file_name == shard_file_name)
        })
        .cloned()
        .collect()
}

/// Validates the complete separate visual-tower sidecar before MLX mapping.
pub(super) fn validate_vision_sidecar(
    required_files: &HashMap<String, ValidatedRequiredFile>,
    vision_tensor_profiles: &[crate::TensorProfile],
) -> Result<(), ArtifactValidationError> {
    let vision_sidecar_file = required_files
        .get(VISION_SIDECAR_FILE_NAME)
        .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
            file_name: VISION_SIDECAR_FILE_NAME.to_owned(),
        })?;
    let accepted_extra_tensor_names = HashSet::new();
    crate::bounded_safetensors::validate_bounded_safetensors_with_partial_profiles(
        vision_sidecar_file.file(),
        vision_sidecar_file.size_bytes(),
        VISION_SIDECAR_FILE_NAME,
        vision_tensor_profiles,
        &accepted_extra_tensor_names,
    )?;
    Ok(())
}
