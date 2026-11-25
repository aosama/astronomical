//! What the shared Qwen 3.5 core reads from a validated folded pack (issue
//! #2473, Prism step 5): the same identity, sizes and limits the Prism engine
//! reads from the pack and its configuration when it is constructed.

use super::ValidatedFoldedPack;
use crate::PersistentVisualEmbeddingModelContract;
use crate::prism_hadamard_qwen35::astro_decoder_state_cache_layout::{
    full_attention_key_value_bytes_per_context_token,
    prism_hadamard_qwen35_astro_decoder_state_cache_layout,
};
use text_generation::generation_loop::serving_artifact::TextServingArtifact;

/// A folded pack stores every weight in one safetensors file, which is the
/// shard count the Prism engine's loading attribution reports.
const FOLDED_PACK_WEIGHT_FILE_COUNT: usize = 1;

impl TextServingArtifact for ValidatedFoldedPack {
    fn model_id(&self) -> &str {
        ValidatedFoldedPack::model_id(self)
    }

    fn revision(&self) -> &str {
        ValidatedFoldedPack::revision(self)
    }

    fn total_payload_bytes(&self) -> u64 {
        ValidatedFoldedPack::total_payload_bytes(self)
    }

    fn shard_count(&self) -> usize {
        FOLDED_PACK_WEIGHT_FILE_COUNT
    }

    /// A folded pack carries text weights only.
    fn persistent_visual_embedding_model_contract(
        &self,
    ) -> Option<PersistentVisualEmbeddingModelContract> {
        None
    }

    fn maximum_position_count(&self) -> u32 {
        self.config().text().max_position_embeddings
    }

    fn vocabulary_size(&self) -> u32 {
        self.config().text().vocab_size
    }

    fn end_of_sequence_token_ids(&self) -> &[u32] {
        &self.config().text().eos_token_ids
    }

    /// The Prism engine reserves no context memory today; this is the
    /// honest per-token figure for when it does: the Float16 key and value
    /// bytes every full-attention layer adds per token, read from the same
    /// layout the live decoder state and the `astro_ssd_prompt_cache` use.
    fn context_memory_reservation_bytes(&self, context_token_count: usize) -> Option<usize> {
        let astro_decoder_state_cache_layout =
            prism_hadamard_qwen35_astro_decoder_state_cache_layout(self.config()).ok()?;
        full_attention_key_value_bytes_per_context_token(&astro_decoder_state_cache_layout)?
            .checked_mul(context_token_count)
    }
}

#[cfg(test)]
#[path = "serving_artifact_tests.rs"]
mod serving_artifact_tests;
