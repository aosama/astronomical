//! Plan-slot expert geometries for K2 Horizon MoVA.
//!
//! Memory sees two routed pools as contiguous `ExpertLayerGeometry` slots:
//! FFN first, then MoVA, per sparse decoder layer. It never hears "MoVA".

use crate::k2_horizon_mova::configuration::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};
use crate::memory::ExpertLayerGeometry;

/// One measured expert-stack payload for a sparse decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct K2HorizonMoVASparseLayerExpertPayload {
    pub decoder_layer_index: usize,
    pub feed_forward_stack_bytes: u64,
    pub mixture_of_values_stack_bytes: u64,
}

/// Builds the contiguous FFN-then-MoVA plan slots for one family member.
pub fn k2_horizon_mova_expert_layer_geometries(
    config: &K2HorizonMoVAConfig,
    sparse_layer_payloads: &[K2HorizonMoVASparseLayerExpertPayload],
) -> Result<Vec<ExpertLayerGeometry>, K2HorizonMoVAExpertGeometryError> {
    let mut geometries = Vec::new();
    for sparse_layer_payload in sparse_layer_payloads {
        match config.layer_kind(sparse_layer_payload.decoder_layer_index) {
            K2HorizonMoVALayerKind::Dense => {
                return Err(K2HorizonMoVAExpertGeometryError::DenseLayer {
                    decoder_layer_index: sparse_layer_payload.decoder_layer_index,
                });
            }
            K2HorizonMoVALayerKind::SparseFeedForward
            | K2HorizonMoVALayerKind::SparseMixtureOfValues => {}
        }
        push_pool_geometry(
            &mut geometries,
            sparse_layer_payload.feed_forward_stack_bytes,
            config.num_experts(),
            config.num_experts_per_tok(),
        )?;
        if config.layer_kind(sparse_layer_payload.decoder_layer_index)
            == K2HorizonMoVALayerKind::SparseMixtureOfValues
            && config.mova_num_experts() > 0
        {
            push_pool_geometry(
                &mut geometries,
                sparse_layer_payload.mixture_of_values_stack_bytes,
                config.mova_num_experts(),
                config.mova_num_experts_per_tok(),
            )?;
        }
    }
    if geometries.is_empty() {
        return Err(K2HorizonMoVAExpertGeometryError::NoSparsePools);
    }
    Ok(geometries)
}

fn push_pool_geometry(
    geometries: &mut Vec<ExpertLayerGeometry>,
    stack_bytes: u64,
    expert_capacity: usize,
    experts_per_token: usize,
) -> Result<(), K2HorizonMoVAExpertGeometryError> {
    if expert_capacity == 0 || experts_per_token == 0 || stack_bytes == 0 {
        return Err(K2HorizonMoVAExpertGeometryError::ZeroPool);
    }
    let expert_capacity_bytes = u64::try_from(expert_capacity)
        .map_err(|_| K2HorizonMoVAExpertGeometryError::ByteCountOverflow)?;
    if stack_bytes % expert_capacity_bytes != 0 {
        return Err(K2HorizonMoVAExpertGeometryError::StackNotDivisible {
            stack_bytes,
            expert_capacity,
        });
    }
    let expert_payload_bytes = stack_bytes / expert_capacity_bytes;
    geometries.push(ExpertLayerGeometry {
        layer_index: geometries.len(),
        complete_layer_payload_bytes: stack_bytes,
        expert_payload_bytes,
        expert_capacity,
        experts_per_token,
    });
    Ok(())
}

/// Failures while mapping family knobs onto plan-slot geometry.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum K2HorizonMoVAExpertGeometryError {
    #[error("dense decoder layer {decoder_layer_index} cannot contribute expert plan slots")]
    DenseLayer { decoder_layer_index: usize },
    #[error("K2 Horizon MoVA expert stacks must be positive")]
    ZeroPool,
    #[error(
        "K2 Horizon MoVA expert stack {stack_bytes} bytes is not divisible by {expert_capacity} experts"
    )]
    StackNotDivisible {
        stack_bytes: u64,
        expert_capacity: usize,
    },
    #[error("K2 Horizon MoVA expert byte arithmetic overflowed")]
    ByteCountOverflow,
    #[error("K2 Horizon MoVA has no sparse expert pools to plan")]
    NoSparsePools,
}
