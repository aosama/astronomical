//! Analytic gradients and test-only weight accessors for the predictor.

use super::network::{ExpertRoutePredictor, LayerHeadGradients};
use super::ops::{backward_head, head_input_vector};
use crate::qwen3_5_moe::expert_paging::route_observation::RouteObservationRecord;

impl ExpertRoutePredictor {
    /// Computes every head gradient and the shared embedding gradient for one
    /// record without applying any update.
    pub(super) fn record_gradients(
        &self,
        observation: &RouteObservationRecord,
    ) -> (f32, Vec<Option<LayerHeadGradients>>, Vec<f32>) {
        let embedding_row = self.embedding_row(observation.input_token_id);
        let mut embedding_gradient = vec![0.0_f32; self.config.embedding_dim];
        let mut total_loss = 0.0_f32;
        let mut gradients_per_layer = Vec::with_capacity(self.config.layer_count);
        for (layer_index, layer_label) in observation.token_route.iter().enumerate() {
            let Some(routed_expert_ids) = layer_label else {
                gradients_per_layer.push(None);
                continue;
            };
            let Some(head) = self.layer_heads.get(layer_index) else {
                gradients_per_layer.push(None);
                continue;
            };
            let head_input = head_input_vector(
                &self.config,
                embedding_row,
                observation
                    .previous_token_route
                    .as_ref()
                    .and_then(|route| route.get(layer_index))
                    .and_then(|maybe_layer_route| maybe_layer_route.as_ref()),
            );
            let (loss, gradients) =
                backward_head(&self.config, head, &head_input, routed_expert_ids);
            total_loss += loss;
            for (embedding_slot, input_gradient) in embedding_gradient.iter_mut().zip(
                gradients
                    .input_gradient
                    .iter()
                    .take(self.config.embedding_dim),
            ) {
                *embedding_slot += input_gradient;
            }
            gradients_per_layer.push(Some(gradients));
        }
        (total_loss, gradients_per_layer, embedding_gradient)
    }

    /// Test-only analytic gradient of one input-weight element.
    #[doc(hidden)]
    #[must_use]
    pub fn input_weight_gradient_for_tests(
        &self,
        observation: &RouteObservationRecord,
        layer_index: usize,
        element_index: usize,
    ) -> f32 {
        self.record_gradients(observation)
            .1
            .into_iter()
            .nth(layer_index)
            .flatten()
            .and_then(|gradients| gradients.input_weights.get(element_index).copied())
            .unwrap_or(0.0)
    }

    /// Test-only analytic gradient of one output-weight element.
    #[doc(hidden)]
    #[must_use]
    pub fn output_weight_gradient_for_tests(
        &self,
        observation: &RouteObservationRecord,
        layer_index: usize,
        element_index: usize,
    ) -> f32 {
        self.record_gradients(observation)
            .1
            .into_iter()
            .nth(layer_index)
            .flatten()
            .and_then(|gradients| gradients.output_weights.get(element_index).copied())
            .unwrap_or(0.0)
    }

    /// Test-only analytic gradient of one embedding element.
    #[doc(hidden)]
    #[must_use]
    pub fn embedding_gradient_for_tests(
        &self,
        observation: &RouteObservationRecord,
        element_index: usize,
    ) -> f32 {
        self.record_gradients(observation)
            .2
            .get(element_index)
            .copied()
            .unwrap_or(0.0)
    }

    /// Test-only weight nudge for the central-difference reference.
    #[doc(hidden)]
    pub fn perturb_embedding_for_tests(&mut self, token_id: u32, element_index: usize, delta: f32) {
        let row_offset =
            usize::try_from(token_id).unwrap_or(usize::MAX) * self.config.embedding_dim;
        if let Some(weight) = self.embedding_table.get_mut(row_offset + element_index) {
            *weight += delta;
        }
    }

    /// Test-only read of one embedding row.
    #[doc(hidden)]
    #[must_use]
    pub fn embedding_row_for_tests(&self, token_id: u32) -> Vec<f32> {
        self.embedding_row(token_id).to_vec()
    }

    /// Test-only weight nudge for one head's input projection.
    #[doc(hidden)]
    pub fn perturb_layer_input_weight_for_tests(
        &mut self,
        layer_index: usize,
        element_index: usize,
        delta: f32,
    ) {
        if let Some(weight) = self
            .layer_heads
            .get_mut(layer_index)
            .and_then(|head| head.input_weights.get_mut(element_index))
        {
            *weight += delta;
        }
    }

    /// Test-only weight nudge for one head's output projection.
    #[doc(hidden)]
    pub fn perturb_layer_output_weight_for_tests(
        &mut self,
        layer_index: usize,
        element_index: usize,
        delta: f32,
    ) {
        if let Some(weight) = self
            .layer_heads
            .get_mut(layer_index)
            .and_then(|head| head.output_weights.get_mut(element_index))
        {
            *weight += delta;
        }
    }
}
