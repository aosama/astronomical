use crate::DecoderCacheLayout;

/// Architecture-neutral decoder-cache contract derived from validated model metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheModelContract {
    model_id: String,
    model_revision: String,
    decoder_cache_layout: DecoderCacheLayout,
}

impl PersistentPromptCacheModelContract {
    /// Binds a validated decoder-state layout to one exact model artifact.
    #[must_use]
    pub fn new(
        model_id: String,
        model_revision: String,
        decoder_cache_layout: DecoderCacheLayout,
    ) -> Self {
        Self {
            model_id,
            model_revision,
            decoder_cache_layout,
        }
    }

    /// Returns the validated model ID bound to this decoder-cache namespace.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the validated model revision bound to this decoder-cache namespace.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// Returns the validated architecture-neutral decoder-state contract.
    #[must_use]
    pub const fn decoder_cache_layout(&self) -> &DecoderCacheLayout {
        &self.decoder_cache_layout
    }
}
