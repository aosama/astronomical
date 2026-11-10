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
    /// A decision model; the wire name matches the discovery model_type.
    Laya,
    /// A decision model; the wire name matches the discovery family.
    Clef,
}

impl DownloadCatalogFamily {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        astronomical_config::exhaustive_table!(self, {
            Self::Qwen3_5 => "qwen3_5",
            Self::Flux2Klein => "flux2_klein",
            Self::QwenImage21 => "qwen_image_21",
            Self::ModernBert => "modernbert",
            Self::PrismHadamardQwen35 => "prism_hadamard_qwen35",
            Self::Laya => "laya",
            Self::Clef => "clef",
        })
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
    /// Answers typed questions with a probability per allowed answer. It
    /// generates no text, so it is not offered for chat.
    Decision,
}

impl DownloadCatalogRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        astronomical_config::exhaustive_table!(self, {
            Self::Chat => "chat",
            Self::Coding => "coding",
            Self::ImageGeneration => "image_generation",
            Self::Helper => "helper",
            Self::Decision => "decision",
        })
    }

    /// Whether models of this role work for other programs and features
    /// instead of being used directly. Nobody starts with one, so one is
    /// never a Get started pick.
    #[must_use]
    pub const fn serves_other_programs(self) -> bool {
        matches!(self, Self::Helper | Self::Decision)
    }
}
