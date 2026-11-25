//! What the shared generation loop reads from a validated model artifact
//! before and while it loads (issue #2473): the limits the engine derives at
//! construction, the identity it reports, and the vision configuration.
//!
//! The loop asks this trait and never a family's own artifact type, so an
//! engine whose artifact is validated differently uses the same core. Each
//! family's artifact answers from its own configuration.

use crate::PersistentVisualEmbeddingModelContract;

pub trait TextServingArtifact {
    fn model_id(&self) -> &str;

    fn revision(&self) -> &str;

    /// The bytes of every weight file the artifact holds.
    fn total_payload_bytes(&self) -> u64;

    fn shard_count(&self) -> usize;

    /// The geometry of the vision tower's embeddings, which checks both a
    /// caller's own visual embeddings and the ones the
    /// `astro_ssd_prompt_cache` stores; absent for a text-only artifact.
    fn persistent_visual_embedding_model_contract(
        &self,
    ) -> Option<PersistentVisualEmbeddingModelContract>;

    /// The model's own position range, the hard limit of a request.
    fn maximum_position_count(&self) -> u32;

    fn vocabulary_size(&self) -> u32;

    fn end_of_sequence_token_ids(&self) -> &[u32];

    /// The memory `context_token_count` tokens of context reserve, or `None`
    /// when the product overflows.
    fn context_memory_reservation_bytes(&self, context_token_count: usize) -> Option<usize>;
}
