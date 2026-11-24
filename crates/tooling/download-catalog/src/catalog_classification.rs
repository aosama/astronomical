//! How a catalog entry is classified: the executable family that decides which
//! engine loads it, and the role that decides where the Models page offers it.

use config::ModelFamily;
use model_family_registry::model_family_named;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

/// The executable family that decides which engine loads a catalog entry:
/// a family the registry names, read from the entry's family identifier.
/// Naming the registry, not a list of its own, keeps a new family to its own
/// crate, one registry line and its catalog entries (issue #2473).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownloadCatalogFamily(ModelFamily);

impl DownloadCatalogFamily {
    /// The registered family with this identifier.
    #[must_use]
    pub fn named(identifier: &str) -> Option<Self> {
        model_family_named(identifier).map(Self)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0.as_str()
    }
}

impl<'de> Deserialize<'de> for DownloadCatalogFamily {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let family_identifier = String::deserialize(deserializer)?;
        Self::named(&family_identifier).ok_or_else(|| {
            D::Error::custom(format!(
                "unknown model family `{family_identifier}`; the family registry names every supported family"
            ))
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
    /// Reads text aloud. It generates no text, so it is not offered for chat.
    Speech,
}

impl DownloadCatalogRole {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        config::exhaustive_table!(self, {
            Self::Chat => "chat",
            Self::Coding => "coding",
            Self::ImageGeneration => "image_generation",
            Self::Helper => "helper",
            Self::Decision => "decision",
            Self::Speech => "speech",
        })
    }

    /// Whether models of this role work for other programs and features
    /// instead of being used directly. Nobody starts with one, so one is
    /// never a Get started pick.
    #[must_use]
    pub const fn serves_other_programs(self) -> bool {
        matches!(self, Self::Helper | Self::Decision | Self::Speech)
    }
}
