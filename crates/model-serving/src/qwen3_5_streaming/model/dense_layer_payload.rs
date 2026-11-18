//! What a dense model's layers occupy: the text weights in memory, the
//! slots its streamed layers use, and the kept layers the memory governor
//! may have given back (issues #1985 and #2221).

use super::Qwen3_5Model;
use super::dense_layer_streamer::Qwen3_5DenseLayerStreamer;

impl Qwen3_5Model {
    /// The text weight bytes in memory: the bound tensors less any dense
    /// decoder layers that are read per forward rather than kept.
    pub(crate) fn resident_model_payload_byte_count(&self) -> u64 {
        let bound_payload_bytes = self
            .base
            .resident_model_payload_byte_count()
            .saturating_add(self.hosted_drafter_payload_bytes.get());
        match &self.dense_layer_streamer {
            Some(dense_layer_streamer) => bound_payload_bytes
                .saturating_sub(dense_layer_streamer.streamed_layer_payload_bytes()),
            None => bound_payload_bytes,
        }
    }

    /// Kept weights the memory governor's capacity covers: retained experts,
    /// plus a dense model's kept layers, which stream once given back.
    pub(crate) fn reclaimable_retained_payload_bytes(&self) -> u64 {
        self.astro_expert_cache_statistics()
            .resident_payload_byte_count
            .saturating_add(self.dense_kept_layer_payload_bytes())
    }

    /// Where this model's decoder layers live, for status; `None` for a
    /// sparse model.
    pub(crate) fn dense_layer_residency_report(
        &self,
    ) -> Option<astronomical_ipc_protocol::WorkerDenseLayerResidency> {
        let dense_layer_streamer = self.dense_layer_streamer.as_ref()?;
        Some(astronomical_ipc_protocol::WorkerDenseLayerResidency {
            decoder_layer_count: u32::try_from(dense_layer_streamer.layer_payload_bytes().len())
                .unwrap_or(u32::MAX),
            kept_layer_count: u32::try_from(dense_layer_streamer.kept_layer_count())
                .unwrap_or(u32::MAX),
            streamed_layer_payload_bytes: dense_layer_streamer.streamed_layer_payload_bytes(),
        })
    }

    /// Payload of the dense layers kept resident.
    pub(crate) fn dense_kept_layer_payload_bytes(&self) -> u64 {
        self.dense_layer_streamer
            .as_ref()
            .map_or(0, Qwen3_5DenseLayerStreamer::kept_layer_payload_bytes)
    }

    /// Gives back kept dense layers until `required_bytes` are freed,
    /// returning the bytes freed; those layers stream until a re-plan.
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
