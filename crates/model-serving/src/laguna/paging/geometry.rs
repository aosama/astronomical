use crate::memory::rotating_prefill_transient_token_count;
use crate::memory::{
    CompleteResidencyDecision, CompleteResidencyRequirements, CurrentExpertLayerResidency,
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlan, plan_phase_aware_expert_residency,
    required_complete_residency_activation_headroom_bytes,
};

use super::error::LagunaPagingError;
use super::layer_plan::LagunaExpertPagingPlan;

/// Descriptor-derived request headroom that admission can charge without laptop constants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaRequestMemoryRequirements {
    complete_expert_payload_bytes: u64,
    complete_prefill_page_bytes: u64,
    routed_decode_page_bytes: u64,
    sliding_prefill_transient_token_count: u32,
}

impl LagunaRequestMemoryRequirements {
    /// Returns the sum of every complete sparse layer.
    #[must_use]
    pub const fn complete_expert_payload_bytes(&self) -> u64 {
        self.complete_expert_payload_bytes
    }

    /// Returns the largest complete sparse-layer page that one prefill step can own.
    #[must_use]
    pub const fn complete_prefill_page_bytes(&self) -> u64 {
        self.complete_prefill_page_bytes
    }

    /// Returns the sum of one contract top-K page per sparse layer.
    #[must_use]
    pub const fn routed_decode_page_bytes(&self) -> u64 {
        self.routed_decode_page_bytes
    }

    /// Returns `window + chunk - 1` tokens for a sliding prefill transient.
    #[must_use]
    pub const fn sliding_prefill_transient_token_count(&self) -> u32 {
        self.sliding_prefill_transient_token_count
    }
}

/// Delegates sliding prefill transient length to the family-neutral #99 helper.
pub fn laguna_sliding_prefill_transient_token_count(
    window_token_count: u32,
    chunk_token_count: u32,
) -> Result<u32, LagunaPagingError> {
    rotating_prefill_transient_token_count(window_token_count, chunk_token_count)
        .map_err(|_| LagunaPagingError::InvalidSlidingTransient)
}

impl LagunaExpertPagingPlan {
    /// Collects complete-layer, routed-page, and sliding-transient charges.
    pub fn request_memory_requirements(
        &self,
        sliding_window_token_count: u32,
        prefill_chunk_token_count: u32,
    ) -> Result<LagunaRequestMemoryRequirements, LagunaPagingError> {
        let mut complete_expert_payload_bytes = 0_u64;
        let mut complete_prefill_page_bytes = 0_u64;
        let mut routed_decode_page_bytes = 0_u64;
        for sparse_layer in self.sparse_layers() {
            let complete_layer_payload_bytes = sparse_layer.complete_layer_payload_byte_count()?;
            complete_expert_payload_bytes = complete_expert_payload_bytes
                .checked_add(complete_layer_payload_bytes)
                .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: sparse_layer.decoder_layer_index(),
                })?;
            complete_prefill_page_bytes =
                complete_prefill_page_bytes.max(complete_layer_payload_bytes);
            routed_decode_page_bytes =
                routed_decode_page_bytes.max(sparse_layer.routed_page_payload_byte_count()?);
        }
        Ok(LagunaRequestMemoryRequirements {
            complete_expert_payload_bytes,
            complete_prefill_page_bytes,
            routed_decode_page_bytes,
            sliding_prefill_transient_token_count: laguna_sliding_prefill_transient_token_count(
                sliding_window_token_count,
                prefill_chunk_token_count,
            )?,
        })
    }

    /// Plans complete-layer versus routed overlay ownership through the existing policy.
    pub fn plan_phase_aware_residency(
        &self,
        phase: ExpertResidencyPhase,
        retained_expert_ceiling_bytes: u64,
        current_residencies: &[CurrentExpertLayerResidency],
    ) -> Result<PhaseAwareExpertResidencyPlan, LagunaPagingError> {
        let layer_geometries = self.layer_geometries()?;
        plan_phase_aware_expert_residency(
            phase,
            retained_expert_ceiling_bytes,
            &layer_geometries,
            current_residencies,
        )
        .map_err(|_| LagunaPagingError::ExpertPayloadOverflow { layer_index: 0 })
    }

    /// Translates Laguna geometry into the centralized replacement-aware residency decision.
    pub fn complete_residency_decision(
        &self,
        current_active_memory_bytes: u64,
        retained_paged_expert_payload_bytes: u64,
        active_memory_ceiling_bytes: u64,
        observed_transient_high_water_bytes: u64,
    ) -> Result<CompleteResidencyDecision, LagunaPagingError> {
        let mut largest_complete_expert_layer_bytes = 0_u64;
        let complete_expert_payload_bytes =
            self.sparse_layers()
                .iter()
                .try_fold(0_u64, |total_payload_bytes, sparse_layer| {
                    let complete_layer_payload_bytes =
                        sparse_layer.complete_layer_payload_byte_count()?;
                    largest_complete_expert_layer_bytes =
                        largest_complete_expert_layer_bytes.max(complete_layer_payload_bytes);
                    total_payload_bytes
                        .checked_add(complete_layer_payload_bytes)
                        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                            layer_index: sparse_layer.decoder_layer_index(),
                        })
                })?;
        let required_activation_headroom_bytes =
            required_complete_residency_activation_headroom_bytes(
                largest_complete_expert_layer_bytes,
                observed_transient_high_water_bytes,
            );
        Ok(CompleteResidencyRequirements {
            current_active_memory_bytes,
            retained_paged_expert_payload_bytes,
            complete_expert_payload_bytes,
            required_headroom_bytes: required_activation_headroom_bytes,
            active_memory_ceiling_bytes,
        }
        .decide())
    }
}
