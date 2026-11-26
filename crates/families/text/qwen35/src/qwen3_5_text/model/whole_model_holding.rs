//! The model held whole in memory, with one routed-expert projection at a
//! time read or replaced: what the quantization sensitivity sweep runs on.
//!
//! The sweep re-encodes one module at a time, forwards calibration samples
//! and puts the original back, so it needs every weight in memory where a
//! swap can reach it. A dense model's decoder layers are therefore bound in
//! the model's own weights rather than handed to the dense layer streamer,
//! and a sparse model's expert layers are seated complete in the
//! `astro_expert_cache`, each expert in the slot of its own id. The forward
//! is the one serving runs, so the sweep measures the kernels a request uses.

use super::Qwen3_5Model;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_text::expert_streaming::RoutedExpertProjection;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5Model {
    /// Holds every weight in memory: a dense model's decoder layers bound in
    /// the model's weights, a sparse model's expert layers seated complete.
    /// Returns whether every expert layer was seated; a model whose experts
    /// do not fit under the limit keeps reading the rest from the SSD, and
    /// its expert projections cannot be replaced.
    pub fn hold_every_layer_in_memory(&mut self) -> Result<bool, TextExecutionError> {
        // Without the streamer the forward reads every decoder layer from
        // the bound weights, which materializing below brings into memory.
        self.dense_layer_streamer = None;
        self.materialize_target_weights()?;
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(true);
        };
        let expert_layer_indexes: Vec<usize> = (0..expert_pager.layer_plans().len()).collect();
        // The tables start with no capacity and only the memory governor
        // raises it while serving. Holding the model whole means room for
        // every complete layer; the caller's MLX limit is the only bound.
        let complete_expert_payload_bytes = expert_pager
            .complete_expert_payload_byte_count()
            .map_err(TextExecutionError::from)?;
        if let Some(astro_expert_cache) = self.astro_expert_cache.as_ref() {
            astro_expert_cache
                .borrow_mut()
                .set_capacity(complete_expert_payload_bytes);
        }
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.seat_complete_layers_before_decode(
            &expert_layer_indexes,
            &mut disabled_performance_attribution,
        )?;
        let mut every_layer_seated = true;
        for &layer_index in &expert_layer_indexes {
            every_layer_seated &= match self.routed_expert_capacity(layer_index) {
                Some(expert_capacity) => {
                    self.has_seated_complete_layer(layer_index, expert_capacity)
                }
                None => false,
            };
        }
        Ok(every_layer_seated)
    }

    /// One routed-expert projection of a layer held complete, across every
    /// expert, as a second reference to the held arrays.
    pub fn routed_expert_projection(
        &self,
        layer_index: usize,
        projection: RoutedExpertProjection,
    ) -> Result<Qwen3_5AffineWeights, TextExecutionError> {
        let (astro_expert_cache, expert_capacity) = self.complete_layer_holder(layer_index)?;
        let astro_expert_cache = astro_expert_cache.borrow();
        let held_projection = astro_expert_cache
            .complete_layer_projection(layer_index, expert_capacity, projection)
            .ok_or(NOT_HELD_COMPLETE)?;
        Ok(held_projection.retained_reference()?)
    }

    /// Puts `replacement` in place of one routed-expert projection across
    /// every expert of a layer held complete, and returns what was there.
    pub fn replace_routed_expert_projection(
        &self,
        layer_index: usize,
        projection: RoutedExpertProjection,
        replacement: Qwen3_5AffineWeights,
    ) -> Result<Qwen3_5AffineWeights, TextExecutionError> {
        let (astro_expert_cache, expert_capacity) = self.complete_layer_holder(layer_index)?;
        astro_expert_cache
            .borrow_mut()
            .replace_complete_layer_projection(
                layer_index,
                expert_capacity,
                projection,
                replacement,
            )
            .or(Err(NOT_HELD_COMPLETE))
    }

    /// The tables holding the layer, with the layer's expert count.
    fn complete_layer_holder(
        &self,
        layer_index: usize,
    ) -> Result<
        (
            &std::cell::RefCell<crate::qwen3_5_text::AstroExpertCache>,
            usize,
        ),
        TextExecutionError,
    > {
        let astro_expert_cache = self.astro_expert_cache.as_ref().ok_or(NOT_HELD_COMPLETE)?;
        let expert_capacity = self
            .routed_expert_capacity(layer_index)
            .ok_or(NOT_HELD_COMPLETE)?;
        Ok((astro_expert_cache, expert_capacity))
    }

    /// How many routed experts the layer has, for a sparse layer.
    fn routed_expert_capacity(&self, layer_index: usize) -> Option<usize> {
        let layer_plan = self.expert_pager.as_ref()?.layer_plans().get(layer_index)?;
        Some(layer_plan.expert_capacity)
    }
}

const NOT_HELD_COMPLETE: TextExecutionError = TextExecutionError::InvalidInput {
    description: "the layer's routed experts are not held complete in memory",
};
