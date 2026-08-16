use std::collections::{BTreeMap, BTreeSet};

use crate::artifact_validation::{
    ArtifactValidationError, TensorDeclarationOrigin, TensorFeature, TensorInventory,
    TensorLocation, TensorSemanticRole, TensorSourceId,
};

use super::Qwen3_5ShardIndex;

/// Builds canonical locations from the Qwen main index without sidecar discovery.
pub(super) fn build_index_tensor_inventory(
    shard_index: &Qwen3_5ShardIndex,
) -> Result<TensorInventory, ArtifactValidationError> {
    let source_id_by_file_name = source_id_by_file_name(shard_index)?;

    let mut inventory = TensorInventory::new();
    let declarations =
        shard_index
            .language_tensor_name_to_shard_file_name()
            .iter()
            .map(|(name, file_name)| (name, file_name, TensorSemanticRole::Target, None))
            .chain(shard_index.mtp_tensor_name_to_shard_file_name().iter().map(
                |(name, file_name)| {
                    (
                        name,
                        file_name,
                        TensorSemanticRole::MultiTokenPrediction,
                        Some(TensorFeature::MultiTokenPrediction),
                    )
                },
            ))
            .chain(
                shard_index
                    .vision_tensor_name_to_shard_file_name()
                    .iter()
                    .map(|(name, file_name)| (name, file_name, TensorSemanticRole::Vision, None)),
            );
    for (canonical_name, file_name, semantic_role, feature) in declarations {
        let source_id = source_id_by_file_name
            .get(file_name)
            .copied()
            .ok_or_else(|| ArtifactValidationError::ProfileMissingRequiredFile {
                file_name: file_name.clone(),
            })?;
        inventory
            .insert(TensorLocation::new(
                canonical_name.clone(),
                canonical_name.clone(),
                source_id,
                semantic_role,
                TensorDeclarationOrigin::MainIndex,
                feature,
            ))
            .map_err(|_| ArtifactValidationError::UnexpectedTensor {
                tensor_name: canonical_name.clone(),
            })?;
    }
    Ok(inventory)
}

pub(super) fn source_id_by_file_name(
    shard_index: &Qwen3_5ShardIndex,
) -> Result<BTreeMap<String, TensorSourceId>, ArtifactValidationError> {
    unique_indexed_file_names(shard_index)
        .into_iter()
        .enumerate()
        .map(|(source_position, file_name)| {
            // Zero remains unused, while u32::MAX is reserved for the architecture sidecar.
            // Return a typed error instead of saturating into either identity on overflow.
            let source_number = u32::try_from(source_position + 1)
                .map_err(|_| ArtifactValidationError::TensorSourceCountOverflow)?;
            if source_number == u32::MAX {
                return Err(ArtifactValidationError::TensorSourceCountOverflow);
            }
            Ok((file_name, TensorSourceId::new(source_number)))
        })
        .collect()
}

fn unique_indexed_file_names(shard_index: &Qwen3_5ShardIndex) -> Vec<String> {
    shard_index
        .language_tensor_name_to_shard_file_name()
        .values()
        .chain(shard_index.mtp_tensor_name_to_shard_file_name().values())
        .chain(shard_index.vision_tensor_name_to_shard_file_name().values())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
