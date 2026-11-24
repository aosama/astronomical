//! What the shared generation loop reads from a validated model artifact
//! before and while it loads (issue #2473): the limits the engine derives at
//! construction, the identity it reports, and the vision configuration.
//!
//! The loop used to hold a `ValidatedQwen3_5Artifact`, which only a Qwen 3.5
//! checkpoint can produce. Asking this trait instead lets an engine whose
//! artifact is validated differently, the folded Prism Hadamard pack, use the
//! same core. The Qwen 3.5 artifact answers from its configuration exactly
//! as the core read it before.

use crate::qwen3_5_core::artifacts::Qwen3_5VisionConfig;
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;

pub trait Qwen3_5ServingArtifact {
    fn model_id(&self) -> &str;

    fn revision(&self) -> &str;

    /// The bytes of every weight file the artifact holds.
    fn total_payload_bytes(&self) -> u64;

    fn shard_count(&self) -> usize;

    /// The vision tower's configuration, absent for a text-only artifact.
    fn vision_config(&self) -> Option<&Qwen3_5VisionConfig>;

    /// The model's own position range, the hard limit of a request.
    fn maximum_position_count(&self) -> u32;

    fn vocabulary_size(&self) -> u32;

    fn end_of_sequence_token_ids(&self) -> &[u32];

    /// The memory `context_token_count` tokens of context reserve, or `None`
    /// when the product overflows.
    fn context_memory_reservation_bytes(&self, context_token_count: usize) -> Option<usize>;
}

impl Qwen3_5ServingArtifact for ValidatedQwen3_5Artifact {
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

    fn vision_config(&self) -> Option<&Qwen3_5VisionConfig> {
        ValidatedQwen3_5Artifact::vision_config(self)
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
