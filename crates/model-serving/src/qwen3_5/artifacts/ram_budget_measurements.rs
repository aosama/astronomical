//! Artifact-side RAM-budget measurements for a validated Qwen3.5 artifact.
//!
//! This adapter reads SafeTensors headers through the existing expert-layer
//! planner and maps the plans into the measured layer facts that
//! `memory/mlx_ram_budget_geometry.rs` consumes. The byte arithmetic itself is
//! owned by `memory`; this file only measures. It does not load weights onto
//! the GPU and does not invent gigabyte souvenirs.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::MlxRamBudgetModelGeometry;
use crate::artifact_validation::TensorDeclarationOrigin;
use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::memory::{
    MeasuredExpertLayerPayload, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
use crate::qwen3_5::{Qwen3_5FeedForwardArchitecture, ValidatedQwen3_5Artifact};
use crate::qwen3_5_moe::expert_paging::quantized_expert_layer_plan::build_quantized_expert_layer_plan_with_stored_names_and_header_cache;

/// Why disk-only RAM measurements could not be composed for this artifact.
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
    let measured_layer_payloads = measured_layer_payloads(&layer_plans)?;
    let complete_residency_transient_bytes = complete_residency_transient_bytes(&layer_plans)?;
    mlx_ram_budget_model_geometry_from_measured_layer_facts(
        &measured_layer_payloads,
        validated_artifact.total_payload_bytes(),
        usize::try_from(validated_artifact.config().experts_per_token()).unwrap_or(usize::MAX),
        complete_residency_transient_bytes,
    )
    .map_err(map_composer_error)
}

fn measured_layer_payloads(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<MeasuredExpertLayerPayload>, Qwen3_5RamBudgetGeometryError> {
    layer_plans
        .iter()
        .map(|layer_plan| {
            Ok(MeasuredExpertLayerPayload::new(
                layer_plan
                    .complete_expert_payload_byte_count()
                    .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?,
                layer_plan.expert_capacity,
            ))
        })
        .collect()
}

/// Transient payload expected to coexist with complete residency at startup.
///
/// With MLX linked, the gate-up fusion transient is the observed worst case;
/// without it, the largest complete layer stands in so hermetic tests still
/// reserve a meaningful floor.
fn complete_residency_transient_bytes(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<u64, Qwen3_5RamBudgetGeometryError> {
    #[cfg(feature = "direct-mlx")]
    {
        crate::qwen3_5_moe::maximum_resident_gate_up_fusion_transient_payload_bytes(layer_plans)
            .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertLayerPlan)
    }
    #[cfg(not(feature = "direct-mlx"))]
    {
        layer_plans
            .iter()
            .try_fold(0_u64, |largest_layer_bytes, layer_plan| {
                Ok(largest_layer_bytes.max(
                    layer_plan
                        .complete_expert_payload_byte_count()
                        .map_err(|_| Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow)?,
                ))
            })
    }
}

fn map_composer_error(composer_error: RamBudgetGeometryError) -> Qwen3_5RamBudgetGeometryError {
    match composer_error {
        RamBudgetGeometryError::NotSparseMixtureOfExperts => {
            Qwen3_5RamBudgetGeometryError::NotSparseMixtureOfExperts
        }
        RamBudgetGeometryError::ExpertPayloadOverflow => {
            Qwen3_5RamBudgetGeometryError::ExpertPayloadOverflow
        }
    }
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
