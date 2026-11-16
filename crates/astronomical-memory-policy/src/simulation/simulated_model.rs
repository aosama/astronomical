//! A model as sizes, with its true MLX working buffers.

use crate::ledger::{LedgerPhase, ModelMemoryShape};

/// Everything the simulated engine needs to know about a model.
#[derive(Clone, Debug)]
pub(crate) struct SimulatedModel {
    pub(crate) core_weight_bytes: u64,
    /// Each streamable layer's bytes, in layer order.
    pub(crate) layer_weight_bytes: Vec<u64>,
    pub(crate) context_bytes_per_token: u64,
    /// Layers in flight while a prompt is read: the one computing and the
    /// one being read ahead.
    pub(crate) reading_stream_slot_count: u64,
    pub(crate) writing_reply_stream_slot_bytes: u64,
    /// What one layer step truly needs per token of its span while reading.
    /// The engine is not told this; it has to measure it.
    pub(crate) true_reading_mlx_working_buffer_bytes_per_span_token: u64,
    pub(crate) true_writing_mlx_working_buffer_bytes: u64,
}

impl SimulatedModel {
    pub(crate) fn largest_layer_weight_bytes(&self) -> u64 {
        let mut largest_layer_weight_bytes = 0;
        for layer_weight_bytes in &self.layer_weight_bytes {
            largest_layer_weight_bytes = largest_layer_weight_bytes.max(*layer_weight_bytes);
        }
        largest_layer_weight_bytes
    }

    pub(crate) fn streamable_weight_bytes(&self) -> u64 {
        let mut streamable_weight_bytes = 0_u64;
        for layer_weight_bytes in &self.layer_weight_bytes {
            streamable_weight_bytes = streamable_weight_bytes.saturating_add(*layer_weight_bytes);
        }
        streamable_weight_bytes
    }

    pub(crate) fn memory_shape(&self) -> ModelMemoryShape {
        ModelMemoryShape {
            core_weight_bytes: self.core_weight_bytes,
            streamable_weight_bytes: self.streamable_weight_bytes(),
            reading_prompt_stream_slot_bytes: self
                .largest_layer_weight_bytes()
                .saturating_mul(self.reading_stream_slot_count),
            writing_reply_stream_slot_bytes: self.writing_reply_stream_slot_bytes,
        }
    }

    /// What one layer step of `phase` truly needs.
    pub(crate) fn true_mlx_working_buffer_bytes(&self, phase: LedgerPhase) -> u64 {
        match phase {
            LedgerPhase::Idle => 0,
            LedgerPhase::WritingReply => self.true_writing_mlx_working_buffer_bytes,
            LedgerPhase::ReadingPrompt { span_token_count } => self
                .true_reading_mlx_working_buffer_bytes_per_span_token
                .saturating_mul(u64::from(span_token_count)),
        }
    }
}
