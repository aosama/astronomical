//! Provenance and license identity for the reviewed Qwen-Image-2.1 artifact.

/// The artifact's official model identifier, as the catalog and provenance record it.
pub const QWEN_IMAGE_21_OFFICIAL_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
/// The provider-prefixed identity the reviewed package is published under.
pub const QWEN_IMAGE_21_PROVIDER_MODEL_ID: &str = "mlx-community/Qwen-Image-2.1-MLX-4bit";
/// The license family the reviewed package carries.
pub const QWEN_IMAGE_21_LICENSE_IDENTIFIER: &str = "qwen-research";

/// Discovery provenance required before local bytes may claim the reviewed profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21ArtifactProvenance {
    model_id: String,
    revision: String,
    license_identifier: String,
}

impl QwenImage21ArtifactProvenance {
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        license_identifier: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            license_identifier: license_identifier.into(),
        }
    }

    pub fn official() -> Self {
        Self::new(
            QWEN_IMAGE_21_PROVIDER_MODEL_ID,
            QWEN_IMAGE_21_OFFICIAL_REVISION,
            QWEN_IMAGE_21_LICENSE_IDENTIFIER,
        )
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    #[must_use]
    pub fn license_identifier(&self) -> &str {
        &self.license_identifier
    }
}

/// License metadata exposed without treating a directory or model name as authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenImage21License;

impl QwenImage21License {
    /// The license family's short identifier, as provenance records it.
    pub const fn identifier(&self) -> &'static str {
        QWEN_IMAGE_21_LICENSE_IDENTIFIER
    }

    /// The human-readable license name.
    pub const fn display_name(&self) -> &'static str {
        "Qwen Research License Agreement"
    }

    /// The page the license terms are published under. The upstream model card hosts the
    /// agreement's text, so it is the durable reference for what this identifier names.
    pub const fn canonical_url(&self) -> &'static str {
        "https://huggingface.co/Qwen/Qwen-Image-2.1"
    }
}

/// The Hub revision of the reviewed snapshot that `official()` records. Content checks, not
/// this SHA, are what validate a package — this constant exists so callers can construct the
/// matching provenance once instead of copying the SHA around.
const QWEN_IMAGE_21_OFFICIAL_REVISION: &str = "4db4e8c0c0e7a1debf0320415bec8388e888494c";
