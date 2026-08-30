//! Disk-only MLX RAM geometry for a validated Qwen3.5 artifact.
//!
//! Reads SafeTensors headers through the existing expert-layer planner. Does not
//! load weights onto the GPU and does not invent gigabyte souvenirs.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::artifact_validation::TensorDeclarationOrigin;
use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::qwen3_5::{Qwen3_5FeedForwardArchitecture, ValidatedQwen3_5Artifact};
use crate::qwen3_5_moe::expert_paging::quantized_expert_layer_plan::build_quantized_expert_layer_plan_with_stored_names_and_header_cache;
use crate::{MlxRamBudgetModelGeometry, required_complete_residency_activation_headroom_bytes};

/// Why disk-only RAM geometry could not be composed for this artifact.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Qwen3_5RamBudgetGeometryError {
    #[error("this Qwen3.5 artifact has no sparse expert payload")]
    NotSparseMixtureOfExperts,
    #[error("expert layer plans could not be read from the validated artifact")]
    ExpertLayerPlan,
    #[error("expert payload accounting overflowed")]
    ExpertPayloadOverflow,
}

/// Returns payload geometry and startup complete-residency headroom for one artifact.
pub fn mlx_ram_budget_model_geometry_from_validated_artifact(
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory: &Path,
) -> Result<(MlxRamBudgetModelGeometry, u64), Qwen3_5RamBudgetGeometryError> {
    if validated_artifact.config().feed_forward_architecture()
        != Qwen3_5FeedForwardArchitecture::MixtureOfExperts
    {
        return Err(Qwen3_5RamBudgetGeometryError::NotSparseMixtureOfExperts);
    }
    let layer_plans =
        expert_layer_plans_from_validated_artifact(validated_artifact, model_directory)?;
    let mut complete_expert_payload_bytes = 0_u64;
    let mut largest_complete_expert_layer_bytes = 0_u64;
    for layer_plan in &layer_plans {
        let layer_payload_bytes = layer_plan
            .complete_expert_payload_byte_count()
            .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?;
        complete_expert_payload_bytes = complete_expert_payload_bytes
            .checked_add(layer_payload_bytes)
            .ok_or(Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?;
        largest_complete_expert_layer_bytes =
            largest_complete_expert_layer_bytes.max(layer_payload_bytes);
    }
    if complete_expert_payload_bytes == 0 {
        return Err(Qwen3_5RamBudgetGeometryError::NotSparseMixtureOfExperts);
    }
    let largest_routed_expert_page_bytes = largest_routed_expert_page_bytes(
        &layer_plans,
        usize::try_from(validated_artifact.config().experts_per_token()).unwrap_or(usize::MAX),
    )?;
    let model_core_payload_bytes = validated_artifact
        .total_payload_bytes()
        .saturating_sub(complete_expert_payload_bytes);
    let required_headroom_bytes = startup_complete_residency_headroom_bytes(&layer_plans)?;
    Ok((
        MlxRamBudgetModelGeometry {
            model_core_payload_bytes,
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
            sequence_state_bytes_per_token: 0,
        },
        required_headroom_bytes,
    ))
}

fn expert_layer_plans_from_validated_artifact(
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_directory: &Path,
) -> Result<Vec<QuantizedExpertLayerPlan>, Qwen3_5RamBudgetGeometryError> {
    let mut tensor_name_to_shard_file_name: HashMap<String, String> = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .chain(
            validated_artifact
                .shard_index()
                .mtp_tensor_name_to_shard_file_name(),
        )
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect();
    if let Some(sidecar_file_name) = validated_artifact.mtp_sidecar_file_name() {
        for location in validated_artifact
            .tensor_inventory()
            .locations()
            .filter(|location| {
                location.declaration_origin() == TensorDeclarationOrigin::ArchitectureSidecar
            })
        {
            tensor_name_to_shard_file_name.insert(
                location.canonical_name().to_owned(),
                sidecar_file_name.to_owned(),
            );
        }
    }
    let stored_tensor_name_by_canonical_name = validated_artifact
        .tensor_inventory()
        .locations()
        .map(|location| {
            (
                location.canonical_name().to_owned(),
                location.stored_name().to_owned(),
            )
        })
        .collect::<HashMap<_, _>>();
    let include_mtp_sparse_expert_layer = tensor_name_to_shard_file_name
        .keys()
        .any(|tensor_name| tensor_name.contains("language_model.mtp.layers.0.mlp.switch_mlp."));
    let decoder_layer_count = validated_artifact.config().layer_count() as usize;
    let mut layer_plans =
        Vec::with_capacity(decoder_layer_count + usize::from(include_mtp_sparse_expert_layer));
    let mut safetensors_header_by_source_file = HashMap::new();
    for decoder_layer_index in 0..decoder_layer_count {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
        let layer_plan = build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
            model_directory,
            &tensor_name_to_shard_file_name,
            &stored_tensor_name_by_canonical_name,
            &layer_prefix,
            validated_artifact.config(),
            &mut safetensors_header_by_source_file,
        )
        .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertLayerPlan)?;
        layer_plans.push(layer_plan);
    }
    if include_mtp_sparse_expert_layer {
        let mtp_layer_plan = build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
            model_directory,
            &tensor_name_to_shard_file_name,
            &stored_tensor_name_by_canonical_name,
            "language_model.mtp.layers.0.mlp",
            validated_artifact.config(),
            &mut safetensors_header_by_source_file,
        )
        .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertLayerPlan)?;
        layer_plans.push(mtp_layer_plan);
    }
    Ok(layer_plans)
}

fn largest_routed_expert_page_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
    experts_per_token: usize,
) -> Result<u64, Qwen3_5RamBudgetGeometryError> {
    layer_plans
        .iter()
        .try_fold(0_u64, |largest_page_bytes, layer_plan| {
            let complete_layer_payload_bytes = layer_plan
                .complete_expert_payload_byte_count()
                .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?;
            let routed_expert_count = experts_per_token.min(layer_plan.expert_capacity);
            let routed_page_bytes = u128::from(complete_layer_payload_bytes)
                .saturating_mul(routed_expert_count as u128)
                / (layer_plan.expert_capacity.max(1) as u128);
            Ok(largest_page_bytes.max(u64::try_from(routed_page_bytes).unwrap_or(u64::MAX)))
        })
}

fn startup_complete_residency_headroom_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, Qwen3_5RamBudgetGeometryError> {
    #[cfg(feature = "direct-mlx")]
    {
        let fusion_transient_payload_bytes =
            crate::qwen3_5_moe::maximum_resident_gate_up_fusion_transient_payload_bytes(
                layer_plans,
            )
            .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertLayerPlan)?;
        Ok(required_complete_residency_activation_headroom_bytes(
            fusion_transient_payload_bytes,
            0,
        ))
    }
    #[cfg(not(feature = "direct-mlx"))]
    {
        let largest_complete_expert_layer_bytes =
            layer_plans
                .iter()
                .try_fold(0_u64, |largest_layer_bytes, layer_plan| {
                    Ok(largest_layer_bytes.max(
                        layer_plan
                            .complete_expert_payload_byte_count()
                            .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?,
                    ))
                })?;
        Ok(required_complete_residency_activation_headroom_bytes(
            largest_complete_expert_layer_bytes,
            0,
        ))
    }
}
