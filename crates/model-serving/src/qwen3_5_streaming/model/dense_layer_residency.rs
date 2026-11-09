//! How many of a dense model's decoder layers stay resident (issue #1985).
//!
//! At load the streaming engine measures the working memory one streamed
//! layer's forward needs, plans how many leading layers fit beside it and the
//! two layer slots under the effective limit, and keeps those. A live limit
//! change and the end of a request plan again; request admission can give
//! kept layers back so they stream rather than refuse a request.

use astronomical_memory_policy::residency::{
    DenseLayerResidencyPlan, DenseLayerResidencyRequirements,
};

use super::Qwen3_5Model;
use super::dense_layer_counts::{KeptLayerReplan, kept_layer_count_after_replan};
use super::dense_layer_streamer::DENSE_STREAMING_LAYER_SLOT_COUNT;
use super::dense_layer_working_memory_probe::DenseLayerProbeShape;
use crate::MlxRamBudgetMeasurement;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::weights::DecoderLayerResidency;

/// How many layers one planning pass keeps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KeptLayerTarget {
    /// The count the plan arrives at, moved as the re-plan allows.
    Planned(KeptLayerReplan),
    /// A count pinned for parity and speed measurements.
    Pinned(usize),
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

    /// Measures working memory, plans the resident layers under the effective
    /// limit and keeps them. `resident_layer_count_override` pins the count
    /// for parity and speed measurements. Returns `None` for a sparse model.
    ///
    /// A probe the limit cannot hold does not fail the load (#1902): the
    /// working memory stays unmeasured, so every plan keeps no layer and the
    /// model streams them all, and MLX and macOS decide the rest.
    #[doc(hidden)]
    pub fn plan_dense_layer_residency(
        &self,
        probe_shape: DenseLayerProbeShape,
        resident_layer_count_override: Option<usize>,
    ) -> Result<Option<DenseLayerResidencyPlan>, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(None);
        };
        match self.measure_dense_layer_working_memory(probe_shape) {
            Ok(working_memory_bytes) => {
                dense_layer_streamer.record_measured_working_memory_bytes(working_memory_bytes);
                self.record_dense_working_memory_in_budget(probe_shape, working_memory_bytes);
            }
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
            }
            Err(probe_error) => return Err(probe_error),
        }
        let kept_layer_target = resident_layer_count_override.map_or(
            KeptLayerTarget::Planned(KeptLayerReplan::MatchPlan),
            KeptLayerTarget::Pinned,
        );
        self.keep_planned_dense_layers(
            self.installed_active_memory_limit_bytes(),
            kept_layer_target,
        )
        .map(Some)
    }

    /// Plans again for `active_memory_ceiling_bytes` and keeps or gives back
    /// layers as `kept_layer_replan` allows. A lower limit plans before the
    /// runtime installs it, so the ceiling is passed rather than read.
    pub(crate) fn replan_kept_dense_layers(
        &self,
        active_memory_ceiling_bytes: u64,
        kept_layer_replan: KeptLayerReplan,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if self.dense_layer_streamer.is_none() {
            return Ok(());
        }
        self.keep_planned_dense_layers(
            active_memory_ceiling_bytes,
            KeptLayerTarget::Planned(kept_layer_replan),
        )
        .map(drop)
    }

    /// Plans again under the installed limit, reading back layers a request
    /// gave up once its context no longer needs their room.
    pub(crate) fn replan_kept_dense_layers_under_installed_limit(
        &self,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.replan_kept_dense_layers(
            self.installed_active_memory_limit_bytes(),
            KeptLayerReplan::MatchPlan,
        )
    }

    pub(crate) fn installed_active_memory_limit_bytes(&self) -> u64 {
        self.runtime.memory_limits().active_memory_limit_bytes() as u64
    }

    fn keep_planned_dense_layers(
        &self,
        active_memory_ceiling_bytes: u64,
        kept_layer_target: KeptLayerTarget,
    ) -> Result<DenseLayerResidencyPlan, Qwen3_5ExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "only a dense model plans its resident layers",
            });
        };
        let measured_working_memory_bytes = dense_layer_streamer.measured_working_memory_bytes();
        let residency_plan = DenseLayerResidencyRequirements {
            always_resident_bytes: self
                .mlx_ram_budget()
                .model_geometry()
                .model_core_payload_bytes,
            decoder_layer_streamable_bytes: dense_layer_streamer.layer_payload_bytes(),
            streaming_slot_count: DENSE_STREAMING_LAYER_SLOT_COUNT,
            working_memory_bytes: measured_working_memory_bytes.unwrap_or(0),
            active_memory_ceiling_bytes,
        }
        .plan();
        let kept_layer_count = match (kept_layer_target, measured_working_memory_bytes) {
            (KeptLayerTarget::Pinned(pinned_kept_layer_count), _) => pinned_kept_layer_count,
            // Without measured working memory no plan is safe to keep.
            (KeptLayerTarget::Planned(_), None) => 0,
            (KeptLayerTarget::Planned(kept_layer_replan), Some(_)) => {
                kept_layer_count_after_replan(
                    residency_plan.resident_layer_count,
                    dense_layer_streamer.kept_layer_count(),
                    kept_layer_replan,
                )
            }
        };
        dense_layer_streamer.keep_layers(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            kept_layer_count,
        )?;
        tracing::info!(
            working_memory_bytes = ?measured_working_memory_bytes,
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
}
