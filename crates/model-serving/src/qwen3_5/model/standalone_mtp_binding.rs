//! Attaches validated standalone MTP sources without merging target source identities.

use std::collections::HashMap;

use crate::expert_paging::RetainedExpertLayerCache;
use crate::qwen3_5::artifacts::Qwen3_5StandaloneMtpBindingParts;
use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpWeights;
use crate::qwen3_5::{Qwen3_5FeedForwardArchitecture, Qwen3_5Model};
use crate::qwen3_5_moe::Qwen3_5RetainedExpertLayer;

use super::Qwen3_5ExecutionError;

pub(crate) fn attach_standalone_mtp_weights(
    model: &mut Qwen3_5Model,
    binding_parts: Qwen3_5StandaloneMtpBindingParts,
) -> Result<(), Qwen3_5ExecutionError> {
    let Qwen3_5StandaloneMtpBindingParts {
        binding_config,
        tensor_inventory,
        source_files,
        model_directory,
        source_file_name_by_source_id,
    } = binding_parts;
    let tensor_name_to_shard_file_name = tensor_inventory
        .locations()
        .map(|tensor_location| {
            let source_file_name = source_file_name_by_source_id
                .get(&tensor_location.source_id())
                .cloned()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "standalone MTP source identity has no shard file",
                })?;
            Ok((
                tensor_location.canonical_name().to_owned(),
                source_file_name,
            ))
        })
        .collect::<Result<HashMap<_, _>, Qwen3_5ExecutionError>>()?;
    let stored_tensor_name_by_canonical_name = tensor_inventory
        .locations()
        .map(|tensor_location| {
            (
                tensor_location.canonical_name().to_owned(),
                tensor_location.stored_name().to_owned(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut standalone_sources = HashMap::with_capacity(source_files.len());
    for (source_id, source_file) in source_files {
        standalone_sources.insert(
            source_id,
            model
                .runtime
                .load_safetensors(source_file.into_file(), None)?,
        );
    }
    let mut mtp_weights =
        Qwen3_5MtpWeights::bind_standalone(&binding_config, &tensor_inventory, standalone_sources)?
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "standalone MTP tensor inventory did not bind",
            })?;
    mtp_weights.repair_raw_normalization_weights(&model.runtime, &model.weights)?;
    if binding_config.feed_forward_architecture()
        == Qwen3_5FeedForwardArchitecture::MixtureOfExperts
    {
        let expert_pager =
            model
                .expert_pager
                .as_mut()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "standalone sparse MTP requires target expert paging",
                })?;
        let expert_pager_geometry = expert_pager.append_standalone_mtp_layer(
            model_directory,
            &tensor_name_to_shard_file_name,
            &stored_tensor_name_by_canonical_name,
            &binding_config,
            usize::try_from(binding_config.experts_per_token()).unwrap_or(usize::MAX),
        )?;
        model.retained_expert_layers = Some(std::cell::RefCell::new(RetainedExpertLayerCache::<
            Qwen3_5RetainedExpertLayer,
        >::new(
            expert_pager.layer_count()
        )));
        let mut model_geometry = model.mlx_ram_budget.borrow().model_geometry();
        model_geometry.complete_expert_payload_bytes =
            expert_pager_geometry.complete_expert_payload_bytes;
        model_geometry.largest_complete_expert_layer_bytes =
            expert_pager_geometry.largest_complete_expert_layer_bytes;
        model_geometry.largest_routed_expert_page_bytes =
            expert_pager_geometry.largest_routed_expert_page_bytes;
        model
            .mlx_ram_budget
            .borrow_mut()
            .update_model_geometry(model_geometry);
    }
    let standalone_mtp_payload_bytes = mtp_weights.payload_byte_count();
    let mut model_geometry = model.mlx_ram_budget.borrow().model_geometry();
    model_geometry.model_core_payload_bytes = model_geometry
        .model_core_payload_bytes
        .saturating_add(standalone_mtp_payload_bytes);
    model
        .mlx_ram_budget
        .borrow_mut()
        .update_model_geometry(model_geometry);
    model.mtp_weights = Some(mtp_weights);
    Ok(())
}

impl Qwen3_5Model {
    /// Attaches one independently validated drafter for explicit direct-model qualification.
    ///
    /// Worker startup uses the same internal path. This seam keeps ignored real-artifact tests
    /// from recreating source ownership or bypassing optional-weight materialization.
    pub fn attach_and_materialize_standalone_mtp(
        &mut self,
        standalone_artifact: crate::ValidatedQwen3_5StandaloneMtpArtifact,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let binding_parts = standalone_artifact.into_binding_parts()?;
        attach_standalone_mtp_weights(self, binding_parts)?;
        if let Err(materialization_error) =
            crate::qwen3_5::multi_token_prediction::materialize_optional_weights(self)
        {
            disable_optional_mtp_weights(self);
            return Err(materialization_error);
        }
        Ok(())
    }
}

/// Drops unusable optional MTP state without disturbing target decoder ownership.
pub(crate) fn disable_optional_mtp_weights(model: &mut Qwen3_5Model) {
    let removed_mtp_payload_bytes = model
        .mtp_weights
        .take()
        .map_or(0, |mtp_weights| mtp_weights.payload_byte_count());
    let mut model_geometry = model.mlx_ram_budget.borrow().model_geometry();
    model_geometry.model_core_payload_bytes = model_geometry
        .model_core_payload_bytes
        .saturating_sub(removed_mtp_payload_bytes);
    if let Some(expert_pager) = model.expert_pager.as_mut() {
        match expert_pager.remove_optional_mtp_layer(
            model.config.layer_count() as usize,
            usize::try_from(model.config.experts_per_token()).unwrap_or(usize::MAX),
        ) {
            Ok(expert_pager_geometry) => {
                model_geometry.complete_expert_payload_bytes =
                    expert_pager_geometry.complete_expert_payload_bytes;
                model_geometry.largest_complete_expert_layer_bytes =
                    expert_pager_geometry.largest_complete_expert_layer_bytes;
                model_geometry.largest_routed_expert_page_bytes =
                    expert_pager_geometry.largest_routed_expert_page_bytes;
                model.retained_expert_layers =
                    Some(std::cell::RefCell::new(RetainedExpertLayerCache::<
                        Qwen3_5RetainedExpertLayer,
                    >::new(
                        expert_pager.layer_count()
                    )));
            }
            Err(expert_pager_error) => tracing::warn!(
                error = %expert_pager_error,
                "optional MTP pager cleanup failed; target decoder plans remain authoritative"
            ),
        }
    }
    model
        .mlx_ram_budget
        .borrow_mut()
        .update_model_geometry(model_geometry);
}
