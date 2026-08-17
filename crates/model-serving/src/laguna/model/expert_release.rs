//! Selective native routed-expert release used by safe resident-to-paged demotion.

use crate::laguna::artifacts::{
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};

use super::weights::LagunaNativeWeights;

impl LagunaNativeWeights {
    /// Shared experts and dense projections remain resident because paging replaces only routed experts.
    pub(super) fn release_routed_experts(&mut self, contract: &LagunaTargetContract) -> u64 {
        let mut released_payload_bytes = 0_u64;
        for layer_descriptor in contract.layers() {
            if !matches!(
                layer_descriptor.feed_forward(),
                LagunaFeedForwardDescriptor::Moe(_)
            ) {
                continue;
            }
            let layer_index = layer_descriptor.layer_index();
            if let Some(fused_gate_up) = self.fused_routed_gate_up.remove(&layer_index) {
                released_payload_bytes =
                    released_payload_bytes.saturating_add(fused_gate_up.payload_byte_count());
            }
            for projection in [
                LagunaExpertProjection::Gate,
                LagunaExpertProjection::Up,
                LagunaExpertProjection::Down,
            ] {
                let routed_projection_id = LagunaTensorId::Layer {
                    layer_index,
                    role: LagunaLayerTensorRole::RoutedExpert(projection),
                    component: LagunaTensorComponent::Weight,
                };
                if let Some(projection_weight) = self.linears.remove(&routed_projection_id) {
                    released_payload_bytes = released_payload_bytes
                        .saturating_add(projection_weight.payload_byte_count());
                }
            }
        }
        released_payload_bytes
    }
}
