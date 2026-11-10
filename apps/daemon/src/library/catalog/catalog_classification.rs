//! How a catalog entry is classified: the executable family that decides which
//! engine loads it, and the role that decides where the Models page offers it.

use serde::Deserialize;

/// Executable model families intentionally supported by catalog version 2.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCatalogFamily {
    Qwen3_5,
    Flux2Klein,
    /// The wire name matches the discovery family, including its underscore before 21.
    #[serde(rename = "qwen_image_21")]
    QwenImage21,
    /// The wire name matches the discovery model_type, which has no underscore.
    #[serde(rename = "modernbert")]
    ModernBert,
    /// Wire name matches the folded ternary pack's `model_type`.
    #[serde(rename = "prism_hadamard_qwen35")]
    PrismHadamardQwen35,
}

impl DownloadCatalogFamily {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3_5 => "qwen3_5",
            Self::Flux2Klein => "flux2_klein",
            Self::QwenImage21 => "qwen_image_21",
            Self::ModernBert => "modernbert",
            Self::PrismHadamardQwen35 => "prism_hadamard_qwen35",
        }
    }
}

/// The job a catalog model is offered for; each role has at most one Get started pick.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCatalogRole {
    Chat,
    Coding,
    ImageGeneration,
    Helper,
}

impl DownloadCatalogRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Coding => "coding",
            Self::ImageGeneration => "image_generation",
            Self::Helper => "helper",
        }
    }
}
