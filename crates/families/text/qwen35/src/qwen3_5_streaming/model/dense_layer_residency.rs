//! How many of a dense model's decoder layers stay resident (issues #1985
//! and #2221).
//!
//! At load the streaming engine measures the MLX working buffers one streamed
//! layer's forward needs. The memory governor then says how many weight
//! bytes may be kept, and the kept layers follow it: any event that lowers
//! the capacity gives trailing layers back at once, and layers are read
//! back only while no request runs.
//!
//! Reading back is limited to what a prompt would leave room for. The
//! capacity is larger while idle than while a prompt is read; filling the
//! idle capacity would have every request give layers back as it starts
//! and read them from the drive again as it ends.

use memory_policy::ledger::KeptWeightCapacity;

use super::Qwen3_5Model;
use super::dense_layer_counts::leading_layer_count_within;
use super::expert_layer_mlx_working_buffer_probe::ProbedPromptLayerMlxWorkingBuffers;
use super::layer_probe_shape::LayerProbeShape;
use crate::qwen3_5_core::model_math::weights::DecoderLayerResidency;
use text_generation::execution_error::TextExecutionError;

/// What probing a dense model's streamed layer found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseLayerProbe {
    /// The MLX working buffers one streamed layer's forward needs; `None` when
    /// the limit could not hold the probe.
    pub mlx_working_buffer_bytes: Option<u64>,
    /// What one streamed forward adds beyond the resident tensors: the
    /// layer slots and the MLX working buffers.
    pub streamed_forward_reserve_bytes: u64,
}

impl Qwen3_5Model {
    /// Materializes the target weights as this engine serves them: a dense
    /// model's decoder layers stay with the streamer, which keeps or reads
    /// them itself.
    pub fn materialize_target_weights(&self) -> Result<(), TextExecutionError> {
        let decoder_layer_residency = match self.dense_layer_streamer {
            Some(_) => DecoderLayerResidency::DenseLayerStreamer,
            None => DecoderLayerResidency::BoundWeights,
        };
        self.base
            .materialize_target_weights_with(decoder_layer_residency)
    }

    /// Materializes the target weights, then measures a dense model's
    /// MLX working buffers at `probe_shape`. The layers are kept once the memory
    /// governor opens and says how many fit.
    pub fn materialize_target_weights_and_probe_dense_layers(
        &self,
        probe_shape: LayerProbeShape,
    ) -> Result<(), TextExecutionError> {
        self.materialize_target_weights()?;
        self.probe_dense_layers(probe_shape, None).map(drop)
    }

    /// Measures a dense model's MLX working buffers. `pinned_kept_layer_count`
    /// keeps exactly that many layers, for parity and speed measurements;
    /// otherwise the kept layers follow the memory governor. Returns `None`
    /// for a sparse model.
    ///
    /// A probe the limit cannot hold does not fail the load (#1902): the
    /// MLX working buffers stay unmeasured, so no layer is kept and the model
    /// streams them all, and MLX and macOS decide the rest.
    #[doc(hidden)]
    pub fn probe_dense_layers(
        &self,
        probe_shape: LayerProbeShape,
        pinned_kept_layer_count: Option<usize>,
    ) -> Result<Option<DenseLayerProbe>, TextExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(None);
        };
        match self.measure_dense_layer_mlx_working_buffers(probe_shape) {
            Ok(mlx_working_buffer_bytes) => {
                dense_layer_streamer
                    .record_measured_mlx_working_buffer_bytes(mlx_working_buffer_bytes);
                self.remember_dense_probe_for_the_governor(probe_shape, mlx_working_buffer_bytes);
            }
            Err(probe_error)
                if probe_error
                    .active_memory_limit_exceeded_evidence()
                    .is_some() =>
            {
                tracing::warn!(
                    error = %probe_error,
                    "the dense MLX working-buffer probe exceeded the limit; every layer will stream"
                );
                self.runtime
                    .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
            }
            Err(probe_error) => return Err(probe_error),
        }
        match pinned_kept_layer_count {
            Some(pinned_kept_layer_count) => dense_layer_streamer.keep_layers(
                &self.runtime,
                &self.config,
                self.weights.raw_hugging_face_norm_offsets_apply(),
                pinned_kept_layer_count,
                self.kept_layer_register_order_kernels(),
            )?,
            None => self.keep_dense_layers_the_governor_has_room_for()?,
        }
        let mlx_working_buffer_bytes = dense_layer_streamer.measured_mlx_working_buffer_bytes();
        Ok(Some(DenseLayerProbe {
            mlx_working_buffer_bytes,
            streamed_forward_reserve_bytes: dense_layer_streamer
                .streaming_slot_bytes()
                .saturating_add(mlx_working_buffer_bytes.unwrap_or(0)),
        }))
    }

    /// Keeps the measurement for the governor, which opens after loading.
    ///
    /// A dense model's reply step runs the same streamed layers through the
    /// same slots, and one token needs fewer MLX working buffers than a prompt
    /// span. Recording the prompt figure for the reply phase too keeps the
    /// kept layers the same in both phases, so none is given back when the
    /// reply begins and read again for the next prompt.
    fn remember_dense_probe_for_the_governor(
        &self,
        probe_shape: LayerProbeShape,
        mlx_working_buffer_bytes: u64,
    ) {
        self.probed_prompt_layer_mlx_working_buffers.set(Some(
            ProbedPromptLayerMlxWorkingBuffers {
                span_token_count: u32::try_from(probe_shape.prefill_chunk_token_count)
                    .unwrap_or(u32::MAX),
                mlx_working_buffer_bytes,
                reply_token_mlx_working_buffer_bytes: mlx_working_buffer_bytes,
            },
        ));
    }

    /// Gives trailing kept layers back until what is kept fits `capacity`.
    /// Those layers stream until layers are read back while idle.
    pub(super) fn give_back_kept_dense_layers_beyond(&self, capacity: KeptWeightCapacity) {
        let kept_layer_payload_bytes = self.dense_kept_layer_payload_bytes();
        if kept_layer_payload_bytes > capacity.bytes() {
            self.release_kept_dense_layers(kept_layer_payload_bytes - capacity.bytes());
        }
    }

    /// Reads back the leading layers the governor has room for while no
    /// request runs: after loading, when a request ends, after the limit is
    /// raised. It never gives a layer back; the governor's capacity does
    /// that. During a request, or for a sparse model, it does nothing.
    pub fn keep_dense_layers_the_governor_has_room_for(&self) -> Result<(), TextExecutionError> {
        let Some(dense_layer_streamer) = &self.dense_layer_streamer else {
            return Ok(());
        };
        let Some(idle_kept_weight_room_bytes) = self.idle_kept_weight_room_bytes() else {
            return Ok(());
        };
        let kept_layer_count = leading_layer_count_within(
            dense_layer_streamer.layer_payload_bytes(),
            idle_kept_weight_room_bytes,
        );
        if kept_layer_count <= dense_layer_streamer.kept_layer_count() {
            return Ok(());
        }
        dense_layer_streamer.keep_layers(
            &self.runtime,
            &self.config,
            self.weights.raw_hugging_face_norm_offsets_apply(),
            kept_layer_count,
            self.kept_layer_register_order_kernels(),
        )?;
        tracing::info!(
            kept_layer_count,
            idle_kept_weight_room_bytes,
            "kept the dense layers the memory governor has room for"
        );
        Ok(())
    }
}
