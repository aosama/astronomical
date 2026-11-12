//! What a catalog entry says its model can do: the document form read from
//! the catalog file, the validated form the rest of the daemon uses, and the
//! rule that turns one into the other.

use serde::Deserialize;

use super::download_catalog::DownloadCatalogError;

/// Human-facing capability badges surfaced from the catalog so users can compare models.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DownloadCatalogCapabilities {
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub supports_tool_calls: bool,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub supports_image_generation: bool,
    pub supports_embeddings: bool,
    pub supports_decisions: bool,
    pub supports_speech: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DownloadCatalogCapabilitiesDocument {
    #[serde(default)]
    pub(super) supports_reasoning: bool,
    #[serde(default)]
    pub(super) supports_vision: bool,
    #[serde(default)]
    pub(super) supports_tool_calls: bool,
    pub(super) context_window: Option<u32>,
    pub(super) max_output_tokens: Option<u32>,
    #[serde(default)]
    pub(super) supports_image_generation: bool,
    #[serde(default)]
    pub(super) supports_embeddings: bool,
    #[serde(default)]
    pub(super) supports_decisions: bool,
    #[serde(default)]
    pub(super) supports_speech: bool,
}

impl DownloadCatalogCapabilitiesDocument {
    /// Whether the document names at least one thing the model can do.
    fn declares_any_capability(&self) -> bool {
        self.supports_reasoning
            || self.supports_vision
            || self.supports_tool_calls
            || self.context_window.is_some()
            || self.max_output_tokens.is_some()
            || self.supports_image_generation
            || self.supports_embeddings
            || self.supports_decisions
            || self.supports_speech
    }
}

/// An absent capabilities document means no badges; a present one must declare
/// at least one badge, otherwise the catalog would advertise an empty model.
pub(super) fn parse_entry_capabilities(
    capabilities_document: Option<&DownloadCatalogCapabilitiesDocument>,
    entry_index: usize,
) -> Result<DownloadCatalogCapabilities, DownloadCatalogError> {
    let Some(capabilities_document) = capabilities_document else {
        return Ok(DownloadCatalogCapabilities::default());
    };
    if capabilities_document.declares_any_capability() {
        Ok(DownloadCatalogCapabilities {
            supports_reasoning: capabilities_document.supports_reasoning,
            supports_vision: capabilities_document.supports_vision,
            supports_tool_calls: capabilities_document.supports_tool_calls,
            context_window: capabilities_document.context_window,
            max_output_tokens: capabilities_document.max_output_tokens,
            supports_image_generation: capabilities_document.supports_image_generation,
            supports_embeddings: capabilities_document.supports_embeddings,
            supports_decisions: capabilities_document.supports_decisions,
            supports_speech: capabilities_document.supports_speech,
        })
    } else {
        Err(DownloadCatalogError::InvalidCapabilities { entry_index })
    }
}
