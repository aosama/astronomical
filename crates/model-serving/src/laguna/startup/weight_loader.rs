//! Loads Laguna weights by canonical tensor ID from retained shard descriptors.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};
use std::collections::{BTreeMap, HashMap};

use crate::artifact_validation::ValidatedWeightsFile;
use crate::laguna::artifacts::{
    LagunaCanonicalTensorAssemblyKind, LagunaGlobalTensorRole, LagunaLayerTensorRole,
    LagunaTensorComponent, LagunaTensorId,
};
use crate::laguna::{
    LagunaExecutionError, LagunaFeedForwardDescriptor, LagunaTargetContract, LagunaTensorContract,
};

/// Loads bindable tensors keyed only by canonical IDs.
pub(in crate::laguna) fn load_laguna_bindable_tensors(
    runtime: &MlxRuntime,
    tensor_contract: &LagunaTensorContract,
    target_contract: &LagunaTargetContract,
    shard_files: BTreeMap<String, ValidatedWeightsFile>,
    load_routed_experts: bool,
) -> Result<HashMap<LagunaTensorId, MlxArray>, LagunaExecutionError> {
    let mut loaded_shards = HashMap::new();
    for (shard_file_name, shard_file) in shard_files {
        let loaded_shard = runtime
            .load_safetensors(shard_file.into_file(), None)
            .map_err(|_| {
                LagunaExecutionError::invalid_geometry(
                    "a Laguna shard could not be loaded through its retained descriptor",
                )
            })?;
        loaded_shards.insert(shard_file_name, loaded_shard);
    }

    let mut tensors = HashMap::new();
    for descriptor in tensor_contract.descriptors().values() {
        if !should_load_tensor(descriptor.tensor_id(), load_routed_experts, target_contract) {
            continue;
        }
        if !matches!(
            descriptor.assembly_kind(),
            LagunaCanonicalTensorAssemblyKind::DirectAlias
                | LagunaCanonicalTensorAssemblyKind::StackedSource
        ) {
            if matches!(
                descriptor.tensor_id(),
                LagunaTensorId::Layer {
                    role: LagunaLayerTensorRole::RoutedExpert(_),
                    ..
                }
            ) {
                continue;
            }
            return Err(LagunaExecutionError::invalid_geometry(
                "Laguna startup cannot assemble this canonical tensor layout yet",
            ));
        }
        let Some(primary_source) = descriptor.sources().first() else {
            return Err(LagunaExecutionError::invalid_geometry(
                "a canonical Laguna tensor is missing its retained source",
            ));
        };
        let loaded_shard = loaded_shards
            .get(primary_source.shard_file_name())
            .ok_or_else(|| {
                LagunaExecutionError::invalid_geometry(
                    "a canonical Laguna tensor refers to an unknown shard",
                )
            })?;
        // Physical lookup uses the retained source name, not a parsed alias.
        let weight = loaded_shard
            .tensor(primary_source.raw_tensor_name())
            .map_err(|_| {
                LagunaExecutionError::invalid_geometry(
                    "a retained Laguna source interval is missing from its shard",
                )
            })?;
        tensors.insert(descriptor.tensor_id(), weight);
    }
    Ok(tensors)
}

fn should_load_tensor(
    tensor_id: LagunaTensorId,
    load_routed_experts: bool,
    target_contract: &LagunaTargetContract,
) -> bool {
    match tensor_id {
        LagunaTensorId::Layer {
            role: LagunaLayerTensorRole::RoutedExpert(_),
            component,
            ..
        } => {
            load_routed_experts
                && is_bindable_linear_component(component)
                && target_contract.layers().iter().any(|layer| {
                    matches!(layer.feed_forward(), LagunaFeedForwardDescriptor::Moe(_))
                })
        }
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::TokenEmbedding,
            component,
        } => is_bindable_linear_component(component),
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::FinalNormalization,
            component,
        }
        | LagunaTensorId::Layer {
            role:
                LagunaLayerTensorRole::InputNormalization
                | LagunaLayerTensorRole::PostAttentionNormalization
                | LagunaLayerTensorRole::AttentionQueryNormalization
                | LagunaLayerTensorRole::AttentionKeyNormalization
                | LagunaLayerTensorRole::RouterCorrectionBias,
            component,
            ..
        } => component == LagunaTensorComponent::Weight,
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::OutputHead,
            component,
        }
        | LagunaTensorId::Layer { component, .. } => is_bindable_linear_component(component),
    }
}

fn is_bindable_linear_component(component: LagunaTensorComponent) -> bool {
    matches!(
        component,
        LagunaTensorComponent::Weight
            | LagunaTensorComponent::Scales
            | LagunaTensorComponent::Biases
    )
}
