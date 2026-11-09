//! How many of a dense model's decoder layers stay resident (issue #1985).
//!
//! At load the streaming engine measures the working memory one streamed
//! layer's forward needs, plans how many leading layers fit beside it and the
//! two layer slots under the effective limit, and keeps those. A live limit
//! change plans again, and request admission can give kept layers back so
//! they stream rather than refuse a request.

use astronomical_memory_policy::residency::{
    DenseLayerResidencyPlan, DenseLayerResidencyRequirements,
};

use super::Qwen3_5Model;
use super::dense_layer_streamer::{DENSE_STREAMING_LAYER_SLOT_COUNT, DenseLayerWeights};
use super::dense_layer_working_memory_probe::DenseLayerProbeShape;
use crate::MlxRamBudgetMeasurement;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::weights::DecoderLayerResidency;

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

    /// Materializes the target weights, then measures a dense model's working
    /// memory at `probe_shape` and keeps the layers that fit beside it.
    pub(crate) fn materialize_target_weights_and_plan_dense_layers(
        &self,
        probe_shape: DenseLayerProbeShape,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.materialize_target_weights()?;
        self.plan_dense_layer_residency(probe_shape, None).map(drop)
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
    ///
    /// A probe the limit cannot hold does not fail the load (#1902): the
    /// model then streams every layer, and MLX and macOS decide the rest.
    #[doc(hidden)]
    pub fn plan_dense_layer_residency(
        &self,
        probe_shape: DenseLayerProbeShape,
        resident_layer_count_override: Option<usize>,
    ) -> Result<Option<DenseLayerResidencyPlan>, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(None);
        };
        let working_memory_bytes = match self.measure_dense_layer_working_memory(probe_shape) {
            Ok(working_memory_bytes) => working_memory_bytes,
            Err(probe_error)
                if probe_error
                    .active_memory_limit_exceeded_evidence()
                    .is_some() =>
            {
                tracing::warn!(
                    error = %probe_error,
                    "the dense working-memory probe exceeded the limit; every layer will stream"
                );
                self.runtime
                    .synchronize_gpu_stream_and_clear_allocator_cache()?;
                return self.keep_planned_dense_layers(Some(0)).map(Some);
            }
            Err(probe_error) => return Err(probe_error),
        };
        dense_layer_streamer.record_measured_working_memory_bytes(working_memory_bytes);
        self.record_dense_working_memory_in_budget(probe_shape, working_memory_bytes);
        self.keep_planned_dense_layers(resident_layer_count_override)
            .map(Some)
    }

    /// Plans again with the working memory measured at load, after a live
    /// limit change, and keeps or gives back layers to match.
    pub(crate) fn replan_kept_dense_layers(&self) -> Result<(), Qwen3_5ExecutionError> {
        if self.dense_layer_streamer.is_none() {
            return Ok(());
        }
        self.keep_planned_dense_layers(None).map(drop)
    }

    fn keep_planned_dense_layers(
        &self,
        resident_layer_count_override: Option<usize>,
    ) -> Result<DenseLayerResidencyPlan, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "only a dense model plans its resident layers",
            });
        };
        let residency_plan = DenseLayerResidencyRequirements {
            always_resident_bytes: self
                .mlx_ram_budget()
                .model_geometry()
                .model_core_payload_bytes,
            decoder_layer_streamable_bytes: dense_layer_streamer.layer_payload_bytes(),
            streaming_slot_count: DENSE_STREAMING_LAYER_SLOT_COUNT,
            working_memory_bytes: dense_layer_streamer.measured_working_memory_bytes(),
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
            working_memory_bytes = dense_layer_streamer.measured_working_memory_bytes(),
            planned_resident_layer_count = residency_plan.resident_layer_count,
            kept_layer_count,
            streaming_floor_bytes = residency_plan.streaming_floor_bytes,
            is_below_streaming_floor = residency_plan.is_below_streaming_floor,
            "dense layer residency planned"
        );
        Ok(residency_plan)
    }

    /// Teaches the RAM budget the measured prefill working memory, so request
    /// admission reserves the same amount the plan left room for.
    fn record_dense_working_memory_in_budget(
        &self,
        probe_shape: DenseLayerProbeShape,
        working_memory_bytes: u64,
    ) {
        self.mlx_ram_budget
            .borrow_mut()
            .record_measurement(MlxRamBudgetMeasurement {
                phase: crate::MemoryPhase::Prefill,
                forward_token_count: probe_shape.prefill_chunk_token_count as u64,
                measured_context_and_activation_bytes: working_memory_bytes,
                observed_activation_headroom_bytes: working_memory_bytes,
                exact_temporary_workspace_bytes: 0,
            });
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

    /// Bytes the layer slots need once any layer streams; zero for a sparse
    /// model. A dense model reserves them even with every layer kept, because
    /// giving one back makes it stream.
    pub(crate) fn dense_streaming_slot_reservation_bytes(&self) -> u64 {
        self.dense_layer_streamer.as_ref().map_or(
            0,
            super::dense_layer_streamer::Qwen3_5DenseLayerStreamer::streaming_slot_bytes,
        )
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
                "gave kept dense layers back to make room"
            );
        }
        released_bytes
    }
}
