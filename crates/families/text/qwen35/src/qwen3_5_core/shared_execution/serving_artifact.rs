//! How a validated Qwen 3.5 artifact answers the text shell's generation
//! loop: its identity and limits from its configuration, and the geometry of
//! its vision tower's embeddings when it has one.

use engine_core::astro_ssd_prompt_cache::PersistentVisualEmbeddingModelContract;
use text_generation::generation_loop::serving_artifact::TextServingArtifact;

use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::vision::{Qwen3_5ImageProcessor, Qwen3_5VisionConfig};

impl TextServingArtifact for ValidatedQwen3_5Artifact {
    fn model_id(&self) -> &str {
        ValidatedQwen3_5Artifact::model_id(self)
    }

    fn revision(&self) -> &str {
        ValidatedQwen3_5Artifact::revision(self)
    }

    fn total_payload_bytes(&self) -> u64 {
        ValidatedQwen3_5Artifact::total_payload_bytes(self)
    }

    fn shard_count(&self) -> usize {
        ValidatedQwen3_5Artifact::shard_count(self)
    }

    fn persistent_visual_embedding_model_contract(
        &self,
    ) -> Option<PersistentVisualEmbeddingModelContract> {
        resolve_visual_embedding_model_contract(
            self.vision_config(),
            self.model_id(),
            self.revision(),
        )
    }

    fn maximum_position_count(&self) -> u32 {
        self.config().maximum_position_count()
    }

    fn vocabulary_size(&self) -> u32 {
        self.config().vocabulary_size()
    }

    fn end_of_sequence_token_ids(&self) -> &[u32] {
        self.config().end_of_sequence_token_ids()
    }

    fn context_memory_reservation_bytes(&self, context_token_count: usize) -> Option<usize> {
        self.config()
            .context_memory_reservation_bytes(context_token_count)
    }
}

/// Pure: builds the in-memory vision tensor geometry contract used to
/// validate both direct visual embeddings and optional disk entries. It
/// creates no storage owner; scanning and all other disk work remain inside
/// the branch where the `astro_ssd_prompt_cache` is enabled.
fn resolve_visual_embedding_model_contract(
    qwen3_5_vision_config: Option<&Qwen3_5VisionConfig>,
    resolved_model_id: &str,
    resolved_model_revision: &str,
) -> Option<PersistentVisualEmbeddingModelContract> {
    qwen3_5_vision_config.map(|vision_config| {
        let qwen3_5_image_processor = Qwen3_5ImageProcessor::from_vision_config(vision_config);
        PersistentVisualEmbeddingModelContract::new(
            resolved_model_id.to_owned(),
            resolved_model_revision.to_owned(),
            vision_config.out_hidden_size() as usize,
            qwen3_5_image_processor.maximum_image_token_count_after_spatial_merge(),
        )
    })
}
