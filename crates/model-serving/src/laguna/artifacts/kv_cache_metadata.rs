use std::collections::BTreeMap;

use ::safetensors::Dtype;

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::{
    LagunaNonExecutableMetadataDescriptor, LagunaTensorSourceRole, LocatedRawTensorDescriptor,
    resolve_sources,
};
use super::exact_storage_validation::validate_scalar_sources;
use super::tensor_id::{
    LagunaAttentionProjection, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
use super::tensor_name_contract::LagunaTensorNameContract;
use crate::laguna::{LagunaStorageDescriptor, LagunaTargetContract};

pub(super) fn collect_fp8_kv_cache_metadata(
    storage: &LagunaStorageDescriptor,
    target_contract: &LagunaTargetContract,
    tensor_name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<Vec<LagunaNonExecutableMetadataDescriptor>, LagunaArtifactValidationError> {
    if !storage.has_fp8_kv_cache() {
        return Ok(Vec::new());
    }
    let mut metadata = Vec::with_capacity(target_contract.layers().len().saturating_mul(2));
    for layer_index in 0..target_contract.layers().len() {
        for (projection, component, source_role) in [
            (
                LagunaAttentionProjection::Key,
                LagunaTensorComponent::AttentionKeyScaleMetadata,
                LagunaTensorSourceRole::AttentionKeyScaleMetadata,
            ),
            (
                LagunaAttentionProjection::Value,
                LagunaTensorComponent::AttentionValueScaleMetadata,
                LagunaTensorSourceRole::AttentionValueScaleMetadata,
            ),
        ] {
            let tensor_id = LagunaTensorId::Layer {
                layer_index,
                role: LagunaLayerTensorRole::Attention(projection),
                component,
            };
            let assembly = tensor_name_contract
                .assemblies()
                .get(&tensor_id)
                .ok_or(LagunaArtifactValidationError::ExpectedTensorMissing { tensor_id })?;
            let sources = resolve_sources(tensor_id, assembly, located_tensors, source_role)?;
            validate_scalar_sources(tensor_id, Dtype::F32, &sources)?;
            metadata.push(LagunaNonExecutableMetadataDescriptor::new(
                tensor_id, sources,
            ));
        }
    }
    Ok(metadata)
}
