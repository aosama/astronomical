//! How many of a dense model's decoder layers stay resident (issue #1985).
//!
//! At load the streaming engine measures the working memory one streamed
//! layer's forward needs, plans how many leading layers fit beside it under
//! the effective limit, and keeps those. Request admission can later give kept
//! layers back so they stream, rather than refuse a request.

use astronomical_memory_policy::residency::{
    DenseLayerResidencyPlan, DenseLayerResidencyRequirements,
};
use mlx_c_rust::{MlxArray, MlxDtype};

use super::Qwen3_5Model;
use super::dense_layer_streamer::DenseLayerWeights;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::weights::DecoderLayerResidency;
use crate::qwen3_5_streaming::model::artifact_loading::DENSE_STREAMING_LAYER_SLOT_COUNT;
use crate::qwen3_5_streaming::model::paged_execution_mode::Qwen3_5MoEPagedExecutionMode;

/// Where a forward stands for the configured layer-group submission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerGroupSubmission {
    pub(crate) completed_layer_count: usize,
    pub(crate) decoder_layer_count: usize,
    pub(crate) graph_submission_layer_interval: usize,
}

impl LayerGroupSubmission {
    /// Whether a completed layer group ends here, short of the final layer.
    const fn is_due(self) -> bool {
        self.graph_submission_layer_interval > 0
            && self
                .completed_layer_count
                .is_multiple_of(self.graph_submission_layer_interval)
            && self.completed_layer_count < self.decoder_layer_count
    }
}

/// Where one forward finds one decoder layer.
pub(crate) enum ForwardLayerWeights<'model> {
    Bound(&'model Qwen3_5DecoderLayerWeights),
    Dense(DenseLayerWeights),
}

impl ForwardLayerWeights<'_> {
    pub(crate) fn decoder_layer_weights(&self) -> &Qwen3_5DecoderLayerWeights {
        match self {
            Self::Bound(bound_layer_weights) => bound_layer_weights,
            Self::Dense(dense_layer_weights) => dense_layer_weights.decoder_layer_weights(),
        }
    }

    /// Whether this layer was read for this forward alone and must submit
    /// on its own so its bytes free once it completes.
    pub(crate) const fn is_streamed(&self) -> bool {
        matches!(self, Self::Dense(dense_layer_weights) if dense_layer_weights.is_streamed())
    }
}

impl Qwen3_5Model {
    /// Materializes the target weights as this engine serves them: a dense
    /// model's decoder layers stay with the streamer, which keeps or reads
    /// them itself.
    pub fn materialize_target_weights(&self) -> Result<(), Qwen3_5ExecutionError> {
        let decoder_layer_residency = match self.dense_layer_streamer {
            Some(_) => DecoderLayerResidency::DenseLayerStreamer,
            None => DecoderLayerResidency::BoundWeights,
        };
        self.base
            .materialize_target_weights_with(decoder_layer_residency)
    }

    /// Materializes the target weights, then plans and keeps a dense model's
    /// resident layers with working memory measured at
    /// `probe_token_count` tokens.
    pub(crate) fn materialize_target_weights_and_plan_dense_layers(
        &self,
        probe_token_count: usize,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.materialize_target_weights()?;
        self.plan_dense_layer_residency(probe_token_count, None)
            .map(drop)
    }

    /// The text weight bytes in memory: the bound tensors less any dense
    /// decoder layers that are read per forward rather than kept.
    pub(crate) fn resident_model_payload_byte_count(&self) -> u64 {
        let bound_payload_bytes = self.base.resident_model_payload_byte_count();
        match &self.dense_layer_streamer {
            Some(dense_layer_streamer) => bound_payload_bytes
                .saturating_sub(dense_layer_streamer.streamed_layer_payload_bytes()),
            None => bound_payload_bytes,
        }
    }

    /// Measures working memory, plans the resident layers under the effective
    /// limit and keeps them. `resident_layer_count_override` pins the count
    /// for parity and speed measurements. Returns `None` for a sparse model.
    #[doc(hidden)]
    pub fn plan_dense_layer_residency(
        &self,
        probe_token_count: usize,
        resident_layer_count_override: Option<usize>,
    ) -> Result<Option<DenseLayerResidencyPlan>, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(None);
        };
        let working_memory_bytes = self.measure_dense_layer_working_memory(probe_token_count)?;
        let residency_plan = DenseLayerResidencyRequirements {
            always_resident_bytes: self
                .mlx_ram_budget()
                .model_geometry()
                .model_core_payload_bytes,
            decoder_layer_streamable_bytes: dense_layer_streamer.layer_payload_bytes(),
            streaming_slot_count: DENSE_STREAMING_LAYER_SLOT_COUNT,
            working_memory_bytes,
            active_memory_ceiling_bytes: self.runtime.memory_limits().active_memory_limit_bytes()
                as u64,
        }
        .plan();
        let kept_layer_count =
            resident_layer_count_override.unwrap_or(residency_plan.resident_layer_count);
        dense_layer_streamer.keep_layers(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            kept_layer_count,
        )?;
        tracing::info!(
            working_memory_bytes,
            planned_resident_layer_count = residency_plan.resident_layer_count,
            kept_layer_count,
            streaming_floor_bytes = residency_plan.streaming_floor_bytes,
            is_below_streaming_floor = residency_plan.is_below_streaming_floor,
            "dense layer residency planned"
        );
        Ok(Some(residency_plan))
    }

    /// The largest transient any layer kind's forward adds at
    /// `probe_token_count` tokens, beyond the layer's own weights.
    ///
    /// One linear-attention and one full-attention layer run a zero input,
    /// because the two kinds allocate differently and memory does not depend
    /// on the values.
    fn measure_dense_layer_working_memory(
        &self,
        probe_token_count: usize,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let mut working_memory_bytes = 0_u64;
        for probe_layer_index in self.probe_layer_indexes() {
            working_memory_bytes = working_memory_bytes
                .max(self.measure_one_layer_working_memory(probe_layer_index, probe_token_count)?);
        }
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        Ok(working_memory_bytes)
    }

    /// The first linear-attention layer and the first full-attention layer.
    fn probe_layer_indexes(&self) -> Vec<usize> {
        let layer_count = self.config.layer_count() as usize;
        let mut probe_layer_indexes = Vec::with_capacity(2);
        for wants_full_attention in [false, true] {
            for layer_index in 0..layer_count {
                if self.config.decoder_layer_is_full_attention(layer_index) == wants_full_attention
                {
                    probe_layer_indexes.push(layer_index);
                    break;
                }
            }
        }
        probe_layer_indexes
    }

    fn measure_one_layer_working_memory(
        &self,
        layer_index: usize,
        probe_token_count: usize,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(0);
        };
        let layer_weights = dense_layer_streamer.read_layer(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            layer_index,
        )?;
        let token_count =
            i32::try_from(probe_token_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "the working-memory probe token count exceeds the MLX range",
            })?;
        let mut request_decoder_state =
            RequestDecoderStateStack::empty_from_config_with_full_attention_kv_state_growth_tokens(
                &self.config,
                token_count,
            )?;
        let probe_input = self.runtime.full(
            &[1, token_count, self.config.hidden_size() as i32],
            0.0,
            MlxDtype::BFloat16,
        )?;
        self.runtime
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        let active_bytes_before = self.runtime.memory_snapshot()?.active_memory_bytes() as u64;
        self.runtime.reset_peak_memory()?;
        let probe_output = self.probe_forward(
            &probe_input,
            token_count,
            layer_index,
            &layer_weights,
            &mut request_decoder_state,
        )?;
        self.runtime.evaluate_arrays(&[&probe_output])?;
        let peak_bytes = self.runtime.memory_snapshot()?.peak_memory_bytes() as u64;
        Ok(peak_bytes
            .saturating_sub(active_bytes_before)
            .saturating_sub(dense_layer_streamer.layer_payload_bytes()[layer_index]))
    }

    fn probe_forward(
        &self,
        probe_input: &MlxArray,
        token_count: i32,
        layer_index: usize,
        layer_weights: &Qwen3_5DecoderLayerWeights,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "the probe decoder state lacks the probed layer",
            },
        )?;
        self.forward_decoder_layer(
            probe_input,
            token_count,
            0,
            layer_index,
            layer_weights,
            layer_model_state,
            None,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            &mut PerformanceAttribution::disabled(),
        )
    }

    /// The weights a forward uses for one decoder layer: the bound weights,
    /// or for a dense model the streamer's kept or freshly read layer.
    pub(crate) fn forward_layer_weights(
        &self,
        layer_index: usize,
    ) -> Result<ForwardLayerWeights<'_>, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return self
                .weights
                .decoder_layer_weights
                .get(layer_index)
                .map(ForwardLayerWeights::Bound)
                .ok_or(Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index });
        };
        Ok(ForwardLayerWeights::Dense(
            dense_layer_streamer.layer_weights(
                &self.runtime,
                &self.config,
                self.weights.raw_hugging_face_norm_offsets_apply(),
                layer_index,
            )?,
        ))
    }

    /// Submits work after a completed decoder layer: always for a streamed
    /// dense layer, otherwise at the configured layer interval.
    pub(crate) fn submit_completed_layer(
        &self,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        hidden_states: &MlxArray,
        previous_streamed_layer_output: &mut Option<MlxArray>,
        layer_group_submission: LayerGroupSubmission,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if forward_layer_weights.is_streamed() {
            return self.submit_streamed_dense_layer(hidden_states, previous_streamed_layer_output);
        }
        if layer_group_submission.is_due() {
            // Intermediate mlx_async_eval commits the completed layer group.
            // Never after the final decoder layer: final normalization and
            // logits extend that same graph and the caller owns the terminal
            // evaluation.
            performance_attribution.measure_operation(
                crate::PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                |_performance_attribution| self.runtime.async_eval_arrays(&[hidden_states]),
            )?;
        }
        Ok(())
    }

    /// Submits a streamed dense layer's output, then waits for the streamed
    /// layer before it, prefill and decode alike.
    ///
    /// Building the graph runs ahead of the GPU, so without the wait every
    /// streamed layer would be read and held before the first one finished.
    /// Waiting bounds the layers in flight to the two slots the plan
    /// reserved: the one computing and the one being read for it.
    pub(crate) fn submit_streamed_dense_layer(
        &self,
        streamed_layer_output: &MlxArray,
        previous_streamed_layer_output: &mut Option<MlxArray>,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.runtime.async_eval_arrays(&[streamed_layer_output])?;
        if let Some(previous_layer_output) =
            previous_streamed_layer_output.replace(streamed_layer_output.retain()?)
        {
            self.runtime.evaluate_arrays(&[&previous_layer_output])?;
        }
        Ok(())
    }

    /// Bytes a streamed forward needs for its layer slots; zero when nothing
    /// streams.
    pub(crate) fn dense_streaming_slot_reservation_bytes(&self) -> u64 {
        match &self.dense_layer_streamer {
            Some(dense_layer_streamer)
                if dense_layer_streamer.kept_layer_count()
                    < dense_layer_streamer.layer_payload_bytes().len() =>
            {
                dense_layer_streamer
                    .largest_layer_payload_bytes()
                    .saturating_mul(DENSE_STREAMING_LAYER_SLOT_COUNT)
            }
            _ => 0,
        }
    }

    /// Payload of the dense layers kept resident, which admission may give back.
    pub(crate) fn dense_kept_layer_payload_bytes(&self) -> u64 {
        self.dense_layer_streamer.as_ref().map_or(
            0,
            super::dense_layer_streamer::Qwen3_5DenseLayerStreamer::kept_layer_payload_bytes,
        )
    }

    /// Gives back kept dense layers until `required_bytes` are freed,
    /// returning the bytes freed; those layers stream from then on.
    pub(crate) fn release_kept_dense_layers(&self, required_bytes: u64) -> u64 {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return 0;
        };
        let released_bytes = dense_layer_streamer.release_trailing_kept_layers(required_bytes);
        if released_bytes > 0 {
            tracing::info!(
                released_bytes,
                kept_layer_count = dense_layer_streamer.kept_layer_count(),
                "gave kept dense layers back to make room for a request"
            );
        }
        released_bytes
    }
}
