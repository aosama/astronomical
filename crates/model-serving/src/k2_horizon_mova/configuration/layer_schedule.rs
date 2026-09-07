//! Dense versus sparse versus MoVA dispatch derived from family knobs.
//!
//! Layer indexes come from `num_hidden_layers`, `mlp_only_layers`, and
//! `decoder_sparse_step`. One 36B card must not be compiled into the schedule.

use super::K2HorizonMoVAConfig;

/// Kind of one decoder layer in this family member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum K2HorizonMoVALayerKind {
    /// Dense GQA, dense SwiGLU, no routed experts.
    Dense,
    /// Sparse FFN without MoVA (dense V).
    SparseFeedForward,
    /// Sparse FFN plus Mixture-of-Value attention.
    SparseMixtureOfValues,
}

impl K2HorizonMoVAConfig {
    /// Returns the layer kind implied by this member's knobs.
    #[must_use]
    pub fn layer_kind(&self, decoder_layer_index: usize) -> K2HorizonMoVALayerKind {
        if !self.is_sparse_layer(decoder_layer_index) {
            return K2HorizonMoVALayerKind::Dense;
        }
        if self.mova_num_experts() == 0 {
            K2HorizonMoVALayerKind::SparseFeedForward
        } else {
            K2HorizonMoVALayerKind::SparseMixtureOfValues
        }
    }

    /// Returns whether this decoder index uses routed FFN experts.
    #[must_use]
    pub fn is_sparse_layer(&self, decoder_layer_index: usize) -> bool {
        if self.num_experts() == 0 || self.decoder_sparse_step() == 0 {
            return false;
        }
        if self.mlp_only_layers().contains(&decoder_layer_index) {
            return false;
        }
        (decoder_layer_index + 1) % self.decoder_sparse_step() == 0
    }
}
