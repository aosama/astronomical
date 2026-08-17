//! Sparse-layer coverage formulas shared by model validation and residency telemetry.

use crate::laguna::artifacts::{
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::laguna::paging::LagunaExpertPagingPlan;

use super::error::LagunaExecutionError;
use super::weights::LagunaNativeWeights;

impl LagunaNativeWeights {
    pub(in crate::laguna) fn has_routed_experts(&self, layer_index: usize) -> bool {
        self.fused_routed_gate_up.contains_key(&layer_index)
            || self.linears.contains_key(&LagunaTensorId::Layer {
                layer_index,
                role: LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
                component: LagunaTensorComponent::Weight,
            })
    }
}

pub(super) fn validate_sparse_coverage(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
    paging_plan: Option<&LagunaExpertPagingPlan>,
) -> Result<(), LagunaExecutionError> {
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) {
            continue;
        }
        let layer_index = layer_descriptor.layer_index();
        if weights.has_routed_experts(layer_index) {
            continue;
        }
        let Some(paging_plan) = paging_plan else {
            return Err(LagunaExecutionError::invalid_geometry(
                "a sparse Laguna layer without resident routed experts requires a paging plan",
            ));
        };
        if paging_plan.sparse_layer_for_decoder(layer_index).is_none() {
            return Err(LagunaExecutionError::invalid_geometry(
                "a sparse Laguna layer is missing from the paging plan",
            ));
        }
    }
    Ok(())
}

pub(super) fn sparse_layer_count(contract: &LagunaTargetContract) -> usize {
    contract
        .layers()
        .iter()
        .filter(|layer_descriptor| {
            matches!(
                layer_descriptor.feed_forward(),
                LagunaFeedForwardDescriptor::Moe(_)
            )
        })
        .count()
}

pub(super) fn sparse_layer_counts(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> (usize, usize) {
    let mut sparse_count = 0_usize;
    let mut resident_count = 0_usize;
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) {
            continue;
        }
        sparse_count = sparse_count.saturating_add(1);
        if weights.has_routed_experts(layer_descriptor.layer_index()) {
            resident_count = resident_count.saturating_add(1);
        }
    }
    (sparse_count, resident_count)
}

pub(super) fn resident_sparse_layer_count(
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> usize {
    sparse_layer_counts(contract, weights).1
}

pub(super) fn resident_complete_payload_bytes(
    paging_plan: &LagunaExpertPagingPlan,
    contract: &LagunaTargetContract,
    weights: &LagunaNativeWeights,
) -> Option<u64> {
    let mut complete_payload_bytes = 0_u64;
    for layer_descriptor in contract.layers() {
        if !matches!(
            layer_descriptor.feed_forward(),
            LagunaFeedForwardDescriptor::Moe(_)
        ) || !weights.has_routed_experts(layer_descriptor.layer_index())
        {
            continue;
        }
        let sparse_layer = paging_plan.sparse_layer_for_decoder(layer_descriptor.layer_index())?;
        complete_payload_bytes = complete_payload_bytes
            .checked_add(sparse_layer.complete_layer_payload_byte_count().ok()?)?;
    }
    Some(complete_payload_bytes)
}
