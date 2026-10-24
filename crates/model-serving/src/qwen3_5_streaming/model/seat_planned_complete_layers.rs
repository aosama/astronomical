//! Enacts `memory/`'s pager-backed decode-handoff seating decision.
//!
//! Selected complete layers are retained in the expert page cache before the
//! first decode token; remaining routes continue to use the pager.

use crate::memory::decode_seating::{DecodeExpertSeating, seat_and_materialize_complete_layers};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::{
    PerformanceAttribution, PerformanceCounter, complete_layer_indexes_required_before_decode,
};

impl Qwen3_5Model {
    /// Complete-layer indexes `memory/` requires seated before decode.
    pub(crate) fn planned_complete_layer_indexes_to_seat_before_decode(&self) -> Vec<usize> {
        self.active_expert_residency_plan
            .borrow()
            .as_ref()
            .map(complete_layer_indexes_required_before_decode)
            .unwrap_or_default()
    }

    fn has_seated_complete_layer(&self, layer_index: usize, expert_capacity: usize) -> bool {
        self.retained_experts
            .as_ref()
            .is_some_and(|retained_experts| {
                retained_experts
                    .borrow()
                    .has_complete_layer(layer_index, expert_capacity)
            })
    }

    /// Loads planned complete layers into the retained cache until leftover fills.
    pub(crate) fn seat_complete_layers_before_decode(
        &self,
        layer_indexes: &[usize],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        if self.expert_pager.is_none() {
            return Ok(0);
        }
        seat_and_materialize_complete_layers(
            &mut ModelDecodeExpertSeating {
                model: self,
                performance_attribution,
            },
            layer_indexes,
        )
    }
}

struct ModelDecodeExpertSeating<'model> {
    model: &'model Qwen3_5Model,
    performance_attribution: &'model mut PerformanceAttribution,
}

impl DecodeExpertSeating for ModelDecodeExpertSeating<'_> {
    type Error = Qwen3_5ExecutionError;

    fn seat_complete_layer(&mut self, layer_index: usize) -> Result<Option<u64>, Self::Error> {
        let expert_pager = self
            .model
            .expert_pager
            .as_ref()
            .expect("seating requires a pager");
        let Some(layer_plan) = expert_pager.layer_plans().get(layer_index) else {
            return Err(crate::qwen3_5_streaming::expert_streaming::expert_pager::ExpertPagingError::InvalidPagingPlan {
                    description: format!("decode seating references missing sparse layer {layer_index}"),
                }.into());
        };
        let expert_capacity = layer_plan.expert_capacity;
        if self
            .model
            .has_seated_complete_layer(layer_index, expert_capacity)
        {
            return Ok(Some(0));
        }
        let (_streamed_weights, streamed_manifest) = self.model.stream_complete_expert_layer(
            expert_pager,
            layer_index,
            2_048,
            expert_capacity,
            expert_capacity,
            true,
            self.performance_attribution,
        )?;
        self.performance_attribution.record_counter(
            PerformanceCounter::ExpertResidencyDecodeSeatingStreamedCompletePayloadBytes,
            streamed_manifest.payload_byte_count,
        );
        Ok(self
            .model
            .has_seated_complete_layer(layer_index, expert_capacity)
            .then_some(streamed_manifest.payload_byte_count))
    }

    fn materialize_complete_layers(&mut self) -> Result<(), Self::Error> {
        self.model.materialize_seated_complete_layers()
    }
}

impl Qwen3_5Model {
    /// Bytes already held as complete layers. Decode leftover publication floors
    /// here so a poisoned activation measurement cannot evict pages this request
    /// already paid to read.
    pub(crate) fn seated_complete_layer_payload_bytes(&self) -> u64 {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return 0;
        };
        let expert_capacity = self
            .expert_pager
            .as_ref()
            .and_then(|expert_pager| expert_pager.layer_plans().first())
            .map_or(0, |layer_plan| layer_plan.expert_capacity);
        retained_experts
            .borrow()
            .complete_layer_payload_bytes(expert_capacity)
    }

    /// Evaluates seated complete-layer weights so the next MLX snapshot includes
    /// them. Decode handoff seating adopts lazy pages; without this eval the first
    /// generate token materializes those pages, admission records them as
    /// activation, and leftover publication evicts the same pages.
    pub(crate) fn materialize_seated_complete_layers(&self) -> Result<(), Qwen3_5ExecutionError> {
        let Some(retained_experts) = self.retained_experts.as_ref() else {
            return Ok(());
        };
        let retained_experts = retained_experts.borrow();
        let mut seated_weight_arrays = Vec::new();
        retained_experts.append_resident_array_references(&mut seated_weight_arrays);
        if seated_weight_arrays.is_empty() {
            return Ok(());
        }
        Ok(self.runtime.evaluate_arrays(&seated_weight_arrays)?)
    }
}
