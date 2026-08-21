//! Strict immutable contract for release-authored model metadata.

use std::collections::BTreeSet;

use serde::Deserialize;
use thiserror::Error;

const DOWNLOAD_CATALOG_SCHEMA_VERSION: u32 = 1;
const MAXIMUM_DOWNLOAD_CATALOG_BYTES: usize = 1_000_000;
const MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT: usize = 1_024;
const MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH: usize = 96;
const MAXIMUM_DISPLAY_NAME_BYTES: usize = 256;
const GIT_COMMIT_SHA_HEX_CHARACTER_COUNT: usize = 40;
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const BUNDLED_DOWNLOAD_CATALOG_JSON: &str =
    include_str!("../../../../registry/download_catalog.json");

/// Validated release-bundled entries in authored presentation order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadCatalog {
    schema_version: u32,
    entries: Vec<DownloadCatalogEntry>,
}

/// One immutable Hugging Face artifact declared public by the release catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadCatalogEntry {
    huggingface_id: String,
    revision: String,
    display_name: String,
    family: DownloadCatalogFamily,
    approximate_size_bytes: u64,
}

/// Executable model families intentionally supported by catalog version 1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCatalogFamily {
    Qwen3_5,
    Laguna,
}

impl DownloadCatalogFamily {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3_5 => "qwen3_5",
            Self::Laguna => "laguna",
        }
    }
}

/// Catalog syntax or semantic validation failure.
#[derive(Debug, Error)]
pub enum DownloadCatalogError {
    #[error("download catalog exceeds the 1000000-byte metadata limit")]
    DocumentTooLarge,
    #[error("download catalog is not valid JSON: {0}")]
    Parse(#[source] serde_json::Error),
    #[error("unsupported download catalog schema version {schema_version}")]
    UnsupportedSchemaVersion { schema_version: u32 },
    #[error("download catalog exceeds the 1024-entry metadata limit")]
    TooManyEntries,
    #[error("download catalog entry {entry_index} has an invalid Hugging Face identity")]
    InvalidHuggingFaceId { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid immutable revision")]
    InvalidRevision { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid display name")]
    InvalidDisplayName { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare approximate_size_bytes between 1 and 9007199254740991"
    )]
    InvalidApproximateSize { entry_index: usize },
    #[error("download catalog entry {entry_index} must declare public: true")]
    ModelNotPublic { entry_index: usize },
    #[error("download catalog contains a duplicate or case-colliding Hugging Face identity")]
    DuplicateHuggingFaceId,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadCatalogDocument {
    schema_version: u32,
    entries: Vec<DownloadCatalogEntryDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadCatalogEntryDocument {
    huggingface_id: String,
    revision: String,
    display_name: String,
    family: DownloadCatalogFamily,
    approximate_size_bytes: u64,
    #[serde(rename = "public")]
    is_public: bool,
}

impl DownloadCatalog {
    /// Parses and validates one complete catalog document.
    pub fn parse_json(catalog_json: &str) -> Result<Self, DownloadCatalogError> {
        // Startup validates this source-authored document before serving; explicit metadata
        // bounds prevent release packaging mistakes from consuming arbitrary laptop resources.
        if catalog_json.len() > MAXIMUM_DOWNLOAD_CATALOG_BYTES {
            return Err(DownloadCatalogError::DocumentTooLarge);
        }
        let catalog_document: DownloadCatalogDocument =
            serde_json::from_str(catalog_json).map_err(DownloadCatalogError::Parse)?;
        if catalog_document.schema_version != DOWNLOAD_CATALOG_SCHEMA_VERSION {
            return Err(DownloadCatalogError::UnsupportedSchemaVersion {
                schema_version: catalog_document.schema_version,
            });
        }
        if catalog_document.entries.len() > MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT {
            return Err(DownloadCatalogError::TooManyEntries);
        }

        let mut normalized_huggingface_ids = BTreeSet::new();
        let mut entries = Vec::with_capacity(catalog_document.entries.len());
        for (entry_index, entry_document) in catalog_document.entries.into_iter().enumerate() {
            validate_huggingface_id(&entry_document.huggingface_id, entry_index)?;
            if !normalized_huggingface_ids
                .insert(entry_document.huggingface_id.to_ascii_lowercase())
            {
                return Err(DownloadCatalogError::DuplicateHuggingFaceId);
            }
            if entry_document.revision.len() != GIT_COMMIT_SHA_HEX_CHARACTER_COUNT
                || !entry_document
                    .revision
                    .bytes()
                    .all(|character| character.is_ascii_hexdigit())
            {
                return Err(DownloadCatalogError::InvalidRevision { entry_index });
            }
            if entry_document.display_name.len() > MAXIMUM_DISPLAY_NAME_BYTES
                || entry_document.display_name.trim().is_empty()
                || entry_document.display_name.chars().any(char::is_control)
            {
                return Err(DownloadCatalogError::InvalidDisplayName { entry_index });
            }
            if entry_document.approximate_size_bytes == 0
                || entry_document.approximate_size_bytes > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
            {
                return Err(DownloadCatalogError::InvalidApproximateSize { entry_index });
            }
            if !entry_document.is_public {
                return Err(DownloadCatalogError::ModelNotPublic { entry_index });
            }
            entries.push(DownloadCatalogEntry {
                huggingface_id: entry_document.huggingface_id,
                revision: entry_document.revision,
                display_name: entry_document.display_name,
                family: entry_document.family,
                approximate_size_bytes: entry_document.approximate_size_bytes,
            });
        }

        Ok(Self {
            schema_version: catalog_document.schema_version,
            entries,
        })
    }

    /// Validates the catalog embedded into the current daemon binary.
    pub fn load_bundled() -> Result<Self, DownloadCatalogError> {
        Self::parse_json(BUNDLED_DOWNLOAD_CATALOG_JSON)
    }

    /// Supplies a validated empty catalog to test-focused application builders.
    #[must_use]
    pub const fn empty_v1() -> Self {
        Self {
            schema_version: DOWNLOAD_CATALOG_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[must_use]
    pub fn entries(&self) -> &[DownloadCatalogEntry] {
        &self.entries
    }

    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl DownloadCatalogEntry {
    #[must_use]
    pub fn huggingface_id(&self) -> &str {
        &self.huggingface_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    #[must_use]
    pub const fn family(&self) -> DownloadCatalogFamily {
        self.family
    }

    #[must_use]
    pub const fn approximate_size_bytes(&self) -> u64 {
        self.approximate_size_bytes
    }
}

fn validate_huggingface_id(
    huggingface_id: &str,
    entry_index: usize,
) -> Result<(), DownloadCatalogError> {
    let mut identity_components = huggingface_id.split('/');
    let organization = identity_components.next().unwrap_or_default();
    let model_name = identity_components.next().unwrap_or_default();
    if identity_components.next().is_some()
        || !is_valid_hugging_face_component(organization)
        || !is_valid_hugging_face_component(model_name)
        || model_name.ends_with(".git")
    {
        return Err(DownloadCatalogError::InvalidHuggingFaceId { entry_index });
    }
    Ok(())
}

fn is_valid_hugging_face_component(component: &str) -> bool {
    let component_bytes = component.as_bytes();
    if component.is_empty()
        || component.len() > MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH
        || component_bytes
            .first()
            .is_some_and(|character| matches!(*character, b'-' | b'.'))
        || component_bytes
            .last()
            .is_some_and(|character| matches!(*character, b'-' | b'.'))
        || component.contains("--")
        || component.contains("..")
    {
        return false;
    }
    component_bytes.iter().copied().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, b'-' | b'_' | b'.')
    })
}
