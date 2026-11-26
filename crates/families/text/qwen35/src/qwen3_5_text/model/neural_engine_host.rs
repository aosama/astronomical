//! The streaming model as a host for Neural Engine prompt reading
//! (issue #2585).
//!
//! A dense model on this engine keeps its decoder layers in the dense layer
//! streamer, not bound in the model base. The programs are compiled from a
//! layer's weights between requests, so the model offers its layers only
//! while the streamer keeps every one in memory. With even one layer read
//! from the SSD for each forward there is no weight to compile from, and
//! prompt reading stays on the GPU.

use crate::qwen3_5_core::model::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::neural_engine::neural_engine_host::{
    LAYER_READ_FAILURE, LayerRead, LayerReader, NeuralEngineHost,
};
use crate::qwen3_5_core::neural_engine::neural_engine_prefill_loading::prepare_neural_engine_prefill;
use crate::qwen3_5_text::model::streaming_model::Qwen3_5TextModel;

impl NeuralEngineHost for Qwen3_5TextModel {
    fn dense_layer_count_in_memory(&self) -> Option<usize> {
        let dense_layer_streamer = self.dense_layer_streamer.as_ref()?;
        let decoder_layer_count = dense_layer_streamer.layer_payload_bytes().len();
        let is_every_layer_kept = decoder_layer_count > 0
            && dense_layer_streamer.kept_layer_count() == decoder_layer_count;
        is_every_layer_kept.then_some(decoder_layer_count)
    }

    fn model_payload_bytes_in_memory(&self) -> u64 {
        let kept_layer_payload_bytes = self
            .dense_layer_streamer
            .as_ref()
            .map_or(0, |dense_layer_streamer| {
                dense_layer_streamer.kept_layer_payload_bytes()
            });
        self.base
            .resident_model_payload_byte_count()
            .saturating_add(kept_layer_payload_bytes)
    }

    fn read_dense_layer(&self, layer_index: usize, read: &mut LayerReader<'_>) -> LayerRead {
        let forward_layer_weights = self
            .forward_layer_weights(layer_index)
            .map_err(|_| LAYER_READ_FAILURE)?;
        if forward_layer_weights.is_streamed() {
            return Err("the decoder layer is read from the SSD for each forward");
        }
        read(forward_layer_weights.decoder_layer_weights());
        Ok(())
    }

    fn model_base(&self) -> &Qwen3_5ModelBase {
        &self.base
    }

    fn model_base_mut(&mut self) -> &mut Qwen3_5ModelBase {
        &mut self.base
    }
}

impl Qwen3_5TextModel {
    /// Builds the opt-in Neural Engine programs through the same steps
    /// serving does, from the layers this model keeps, and waits for them,
    /// so a journey can measure one loaded model with and without. A dense
    /// model builds them only with every layer kept
    /// (`materialize_and_keep_every_dense_layer`).
    pub fn prepare_neural_engine_prefill_matching_production(
        &mut self,
        chunking: &ipc_protocol::WorkerChunkingConfiguration,
        astro_neural_engine_program_cache_directory: Option<&std::path::Path>,
    ) -> ipc_protocol::NeuralEnginePrefillStatus {
        prepare_neural_engine_prefill(
            self,
            chunking,
            astro_neural_engine_program_cache_directory,
            None,
        )
    }

    /// The same steps with a chosen token split, so a journey can sweep it.
    pub fn prepare_neural_engine_prefill_with_token_count(
        &mut self,
        chunking: &ipc_protocol::WorkerChunkingConfiguration,
        neural_engine_token_count: u32,
    ) -> ipc_protocol::NeuralEnginePrefillStatus {
        prepare_neural_engine_prefill(self, chunking, None, Some(neural_engine_token_count))
    }

    /// Decides whether the coming prompt span uses the programs.
    ///
    /// The memory ledger has already set aside everything this model needs
    /// at the limit in force, and what is left over once every weight is
    /// kept is its unassigned room. The programs' memory is outside MLX's,
    /// so a span uses them only while they fit in that room.
    pub fn decide_neural_engine_for_prompt_span(&self, span_token_count: usize) {
        let Some(ledger_reading) = self.latest_memory_ledger_reading() else {
            return;
        };
        let ledger_accounts = ledger_reading.accounts;
        self.base.decide_neural_engine_for_chunk(
            span_token_count,
            ledger_accounts
                .limit_bytes
                .saturating_sub(ledger_accounts.unassigned_bytes),
            ledger_accounts.limit_bytes,
        );
    }
}
