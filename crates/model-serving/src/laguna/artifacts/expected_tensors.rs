use std::collections::{BTreeMap, BTreeSet};

use super::artifact_error::LagunaArtifactValidationError;
use super::tensor_id::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
use super::tensor_storage::LagunaTensorStorageEncoding;
use crate::laguna::{
    LagunaCompressedWeightEncoding, LagunaFeedForwardDescriptor, LagunaGatingKind,
    LagunaStorageDescriptor, LagunaTargetContract,
};

/// Canonical logical shape generated only from normalized target geometry.
pub(super) struct LagunaExpectedTensor {
    pub(super) logical_shape: Vec<usize>,
    pub(super) canonical_module_name: Option<String>,
    pub(super) storage_encoding: LagunaTensorStorageEncoding,
}

pub(super) fn expected_tensors(
    target_contract: &LagunaTargetContract,
) -> Result<BTreeMap<LagunaTensorId, LagunaExpectedTensor>, LagunaArtifactValidationError> {
    let model = target_contract.model();
    let vocabulary_size = dimension(model.vocabulary_size())?;
    let hidden_size = dimension(model.hidden_size())?;
    let mut expected_tensors = BTreeMap::new();
    insert_global(
        &mut expected_tensors,
        target_contract.storage(),
        LagunaGlobalTensorRole::TokenEmbedding,
        vec![vocabulary_size, hidden_size],
    );
    insert_global(
        &mut expected_tensors,
        target_contract.storage(),
        LagunaGlobalTensorRole::FinalNormalization,
        vec![hidden_size],
    );
    if !model.has_tied_embeddings() {
        insert_global(
            &mut expected_tensors,
            target_contract.storage(),
            LagunaGlobalTensorRole::OutputHead,
            vec![vocabulary_size, hidden_size],
        );
    }

    for layer in target_contract.layers() {
        let layer_index = layer.layer_index();
        let attention = layer.attention();
        let query_head_count = dimension(attention.query_head_count())?;
        let key_value_head_count = dimension(attention.key_value_head_count())?;
        let head_dimension = dimension(attention.head_dimension())?;
        let query_projection_size = checked_product(query_head_count, head_dimension)?;
        let key_value_projection_size = checked_product(key_value_head_count, head_dimension)?;
        insert_layer(
            &mut expected_tensors,
            target_contract.storage(),
            layer_index,
            LagunaLayerTensorRole::InputNormalization,
            vec![hidden_size],
        );
        insert_layer(
            &mut expected_tensors,
            target_contract.storage(),
            layer_index,
            LagunaLayerTensorRole::PostAttentionNormalization,
            vec![hidden_size],
        );
        for (projection, shape) in [
            (
                LagunaAttentionProjection::Query,
                vec![query_projection_size, hidden_size],
            ),
            (
                LagunaAttentionProjection::Key,
                vec![key_value_projection_size, hidden_size],
            ),
            (
                LagunaAttentionProjection::Value,
                vec![key_value_projection_size, hidden_size],
            ),
            (
                LagunaAttentionProjection::Output,
                vec![hidden_size, query_projection_size],
            ),
        ] {
            insert_layer(
                &mut expected_tensors,
                target_contract.storage(),
                layer_index,
                LagunaLayerTensorRole::Attention(projection),
                shape,
            );
        }
        insert_layer(
            &mut expected_tensors,
            target_contract.storage(),
            layer_index,
            LagunaLayerTensorRole::AttentionQueryNormalization,
            vec![head_dimension],
        );
        insert_layer(
            &mut expected_tensors,
            target_contract.storage(),
            layer_index,
            LagunaLayerTensorRole::AttentionKeyNormalization,
            vec![head_dimension],
        );
        match attention.gating_kind() {
            LagunaGatingKind::None => {}
            LagunaGatingKind::PerHead => insert_layer(
                &mut expected_tensors,
                target_contract.storage(),
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                vec![query_head_count, hidden_size],
            ),
            LagunaGatingKind::PerElement => insert_layer(
                &mut expected_tensors,
                target_contract.storage(),
                layer_index,
                LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Gate),
                vec![query_projection_size, hidden_size],
            ),
        }
        insert_feed_forward_tensors(
            &mut expected_tensors,
            target_contract.storage(),
            layer_index,
            hidden_size,
            layer.feed_forward(),
        )?;
    }
    validate_affine_override_resolution(target_contract.storage(), &expected_tensors)?;
    Ok(expected_tensors)
}

fn insert_feed_forward_tensors(
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
    storage: &LagunaStorageDescriptor,
    layer_index: usize,
    hidden_size: usize,
    feed_forward: &LagunaFeedForwardDescriptor,
) -> Result<(), LagunaArtifactValidationError> {
    match feed_forward {
        LagunaFeedForwardDescriptor::Dense(dense) => {
            let intermediate_size = dimension(dense.intermediate_size())?;
            insert_projection_set(
                expected_tensors,
                storage,
                layer_index,
                hidden_size,
                intermediate_size,
                LagunaLayerTensorRole::DenseFeedForward,
            );
        }
        LagunaFeedForwardDescriptor::Moe(moe) => {
            let expert_count = dimension(moe.expert_count())?;
            let expert_intermediate_size = dimension(moe.expert_intermediate_size())?;
            insert_layer(
                expected_tensors,
                storage,
                layer_index,
                LagunaLayerTensorRole::Router,
                vec![expert_count, hidden_size],
            );
            for (projection, shape) in [
                (
                    LagunaExpertProjection::Gate,
                    vec![expert_count, expert_intermediate_size, hidden_size],
                ),
                (
                    LagunaExpertProjection::Up,
                    vec![expert_count, expert_intermediate_size, hidden_size],
                ),
                (
                    LagunaExpertProjection::Down,
                    vec![expert_count, hidden_size, expert_intermediate_size],
                ),
            ] {
                insert_layer(
                    expected_tensors,
                    storage,
                    layer_index,
                    LagunaLayerTensorRole::RoutedExpert(projection),
                    shape,
                );
            }
            let shared_intermediate_size = dimension(moe.shared_expert_intermediate_size())?;
            if shared_intermediate_size > 0 {
                insert_projection_set(
                    expected_tensors,
                    storage,
                    layer_index,
                    hidden_size,
                    shared_intermediate_size,
                    LagunaLayerTensorRole::SharedExpert,
                );
            }
        }
    }
    Ok(())
}

fn insert_projection_set<F>(
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
    storage: &LagunaStorageDescriptor,
    layer_index: usize,
    hidden_size: usize,
    intermediate_size: usize,
    role: F,
) where
    F: Fn(LagunaExpertProjection) -> LagunaLayerTensorRole,
{
    for (projection, shape) in [
        (
            LagunaExpertProjection::Gate,
            vec![intermediate_size, hidden_size],
        ),
        (
            LagunaExpertProjection::Up,
            vec![intermediate_size, hidden_size],
        ),
        (
            LagunaExpertProjection::Down,
            vec![hidden_size, intermediate_size],
        ),
    ] {
        insert_layer(
            expected_tensors,
            storage,
            layer_index,
            role(projection),
            shape,
        );
    }
}

fn insert_global(
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
    storage: &LagunaStorageDescriptor,
    role: LagunaGlobalTensorRole,
    logical_shape: Vec<usize>,
) {
    let canonical_module_name = match role {
        LagunaGlobalTensorRole::TokenEmbedding => Some("model.embed_tokens".to_owned()),
        LagunaGlobalTensorRole::FinalNormalization => None,
        LagunaGlobalTensorRole::OutputHead => Some("lm_head".to_owned()),
    };
    insert_components(
        expected_tensors,
        storage,
        canonical_module_name,
        logical_shape,
        |component| LagunaTensorId::Global { role, component },
    );
}

fn insert_layer(
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
    storage: &LagunaStorageDescriptor,
    layer_index: usize,
    role: LagunaLayerTensorRole,
    logical_shape: Vec<usize>,
) {
    let canonical_module_name = canonical_layer_module_name(layer_index, role);
    insert_components(
        expected_tensors,
        storage,
        canonical_module_name,
        logical_shape,
        |component| LagunaTensorId::Layer {
            layer_index,
            role,
            component,
        },
    );
}

fn insert_components<F>(
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
    storage: &LagunaStorageDescriptor,
    canonical_module_name: Option<String>,
    logical_shape: Vec<usize>,
    tensor_id_for_component: F,
) where
    F: Fn(LagunaTensorComponent) -> LagunaTensorId,
{
    let storage_encoding = storage_encoding_for_module(storage, canonical_module_name.as_deref());
    let components: &[LagunaTensorComponent] = match &storage_encoding {
        LagunaTensorStorageEncoding::DirectAffine { .. }
        | LagunaTensorStorageEncoding::SymmetricPackedAffine { .. } => &[
            LagunaTensorComponent::Weight,
            LagunaTensorComponent::Scales,
            LagunaTensorComponent::Biases,
        ],
        LagunaTensorStorageEncoding::NativeNvfp4 { .. } => {
            &[LagunaTensorComponent::Weight, LagunaTensorComponent::Scales]
        }
        _ => &[LagunaTensorComponent::Weight],
    };
    for component in components {
        expected_tensors.insert(
            tensor_id_for_component(*component),
            LagunaExpectedTensor {
                logical_shape: logical_shape.clone(),
                canonical_module_name: canonical_module_name.clone(),
                storage_encoding: storage_encoding.clone(),
            },
        );
    }
}

fn storage_encoding_for_module(
    storage: &LagunaStorageDescriptor,
    canonical_module_name: Option<&str>,
) -> LagunaTensorStorageEncoding {
    let Some(module_name) = canonical_module_name else {
        return LagunaTensorStorageEncoding::Unquantized;
    };
    match storage {
        LagunaStorageDescriptor::DirectAffine(affine) => {
            // Router linears may stay native while the rest of the body is affine.
            if module_name.ends_with(".mlp.gate.proj")
                && !affine.module_overrides().contains_key(module_name)
            {
                return LagunaTensorStorageEncoding::Unquantized;
            }
            LagunaTensorStorageEncoding::DirectAffine {
                profile: affine.profile_for_module(module_name),
            }
        }
        LagunaStorageDescriptor::NativeNvfp4(profile) => {
            if module_name.ends_with(".mlp.gate.proj") {
                LagunaTensorStorageEncoding::Unquantized
            } else {
                LagunaTensorStorageEncoding::NativeNvfp4 { profile: *profile }
            }
        }
        LagunaStorageDescriptor::Compressed(compressed)
            if !module_name.ends_with(".mlp.gate.proj")
                && compressed.applies_to_module(module_name) =>
        {
            match compressed.weight_encoding() {
                LagunaCompressedWeightEncoding::SymmetricPackedAffine(profile) => {
                    LagunaTensorStorageEncoding::SymmetricPackedAffine { profile }
                }
                LagunaCompressedWeightEncoding::TwoLevelNvfp4(profile) => {
                    LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { profile }
                }
                LagunaCompressedWeightEncoding::BlockFp8(profile) => {
                    LagunaTensorStorageEncoding::BlockFp8 {
                        block_row_extent: profile.block_row_extent() as usize,
                        block_column_extent: profile.block_column_extent() as usize,
                    }
                }
            }
        }
        LagunaStorageDescriptor::Unquantized | LagunaStorageDescriptor::Compressed(_) => {
            LagunaTensorStorageEncoding::Unquantized
        }
    }
}

#[cfg(feature = "direct-mlx")]
pub(crate) fn laguna_canonical_module_name(tensor_id: LagunaTensorId) -> Option<String> {
    match tensor_id {
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::TokenEmbedding,
            ..
        } => Some("model.embed_tokens".to_owned()),
        LagunaTensorId::Global {
            role: LagunaGlobalTensorRole::OutputHead,
            ..
        } => Some("lm_head".to_owned()),
        LagunaTensorId::Global { .. } => None,
        LagunaTensorId::Layer {
            layer_index, role, ..
        } => canonical_layer_module_name(layer_index, role),
    }
}

fn canonical_layer_module_name(layer_index: usize, role: LagunaLayerTensorRole) -> Option<String> {
    let layer_prefix = format!("model.layers.{layer_index}");
    let module_suffix = match role {
        LagunaLayerTensorRole::Attention(projection) => match projection {
            LagunaAttentionProjection::Query => "self_attn.q_proj".to_owned(),
            LagunaAttentionProjection::Key => "self_attn.k_proj".to_owned(),
            LagunaAttentionProjection::Value => "self_attn.v_proj".to_owned(),
            LagunaAttentionProjection::Output => "self_attn.o_proj".to_owned(),
            LagunaAttentionProjection::Gate => "self_attn.g_proj".to_owned(),
        },
        LagunaLayerTensorRole::DenseFeedForward(projection) => projection_suffix("mlp", projection),
        LagunaLayerTensorRole::SharedExpert(projection) => {
            projection_suffix("mlp.shared_expert", projection)
        }
        // Routed profiles use the canonical stacked execution owner independent of source packaging.
        LagunaLayerTensorRole::RoutedExpert(projection) => {
            projection_suffix("mlp.switch_mlp", projection)
        }
        LagunaLayerTensorRole::Router => "mlp.gate.proj".to_owned(),
        LagunaLayerTensorRole::InputNormalization
        | LagunaLayerTensorRole::PostAttentionNormalization
        | LagunaLayerTensorRole::AttentionQueryNormalization
        | LagunaLayerTensorRole::AttentionKeyNormalization
        | LagunaLayerTensorRole::RouterCorrectionBias
        | LagunaLayerTensorRole::SharedExpertGate => return None,
    };
    Some(format!("{layer_prefix}.{module_suffix}"))
}

fn projection_suffix(owner: &str, projection: LagunaExpertProjection) -> String {
    let projection_name = match projection {
        LagunaExpertProjection::Gate => "gate_proj",
        LagunaExpertProjection::Up => "up_proj",
        LagunaExpertProjection::Down => "down_proj",
    };
    format!("{owner}.{projection_name}")
}

fn validate_affine_override_resolution(
    storage: &LagunaStorageDescriptor,
    expected_tensors: &BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
) -> Result<(), LagunaArtifactValidationError> {
    let LagunaStorageDescriptor::DirectAffine(affine) = storage else {
        return Ok(());
    };
    let executable_module_names = expected_tensors
        .values()
        .filter_map(|expected_tensor| expected_tensor.canonical_module_name.as_deref())
        .collect::<BTreeSet<_>>();
    for module_name in affine.module_overrides().keys() {
        let resolved_module_count = if executable_module_names.contains(module_name.as_str()) {
            1
        } else {
            0
        };
        if resolved_module_count != 1 {
            return Err(LagunaArtifactValidationError::AffineOverrideResolution {
                module_name: module_name.clone(),
                resolved_module_count,
            });
        }
    }
    Ok(())
}

fn dimension(value: u32) -> Result<usize, LagunaArtifactValidationError> {
    usize::try_from(value).map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)
}

fn checked_product(
    left_dimension: usize,
    right_dimension: usize,
) -> Result<usize, LagunaArtifactValidationError> {
    left_dimension
        .checked_mul(right_dimension)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)
}
