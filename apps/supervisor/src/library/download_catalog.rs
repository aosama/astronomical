//! Strict immutable contract for release-authored model metadata.

use std::collections::BTreeSet;

use serde::Deserialize;
use thiserror::Error;

use super::DownloadPathSelection;

const DOWNLOAD_CATALOG_SCHEMA_VERSION: u32 = 1;
const MAXIMUM_DOWNLOAD_CATALOG_BYTES: usize = 1_000_000;
const MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT: usize = 1_024;
const MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH: usize = 96;
const MAXIMUM_DISPLAY_NAME_BYTES: usize = 256;
const MAXIMUM_DESCRIPTION_BYTES: usize = 512;
const MAXIMUM_QUANTIZATION_LABEL_BYTES: usize = 64;
const MAXIMUM_ARCHITECTURE_SUMMARY_BYTES: usize = 256;
const MAXIMUM_UPSTREAM_LICENSE_BYTES: usize = 128;
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
    description: Option<String>,
    capabilities: DownloadCatalogCapabilities,
    quantization_label: Option<String>,
    architecture_summary: Option<String>,
    upstream_license: Option<String>,
    download_path_selection: DownloadPathSelection,
}

/// Human-facing capability badges surfaced from the catalog so users can compare models.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DownloadCatalogCapabilities {
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub supports_tool_calls: bool,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub supports_image_generation: bool,
}

/// Executable model families intentionally supported by catalog version 1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCatalogFamily {
    Qwen3_5,
    Laguna,
    Flux2Klein,
}

impl DownloadCatalogFamily {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3_5 => "qwen3_5",
            Self::Laguna => "laguna",
            Self::Flux2Klein => "flux2_klein",
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
    #[error("download catalog entry {entry_index} has an invalid description")]
    InvalidDescription { entry_index: usize },
    #[error("download catalog entry {entry_index} has invalid capabilities")]
    InvalidCapabilities { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid quantization label")]
    InvalidQuantizationLabel { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid architecture summary")]
    InvalidArchitectureSummary { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid upstream license")]
    InvalidUpstreamLicense { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare approximate_size_bytes between 1 and 9007199254740991"
    )]
    InvalidApproximateSize { entry_index: usize },
    #[error("download catalog entry {entry_index} must declare public: true")]
    ModelNotPublic { entry_index: usize },
    #[error("download catalog contains a duplicate or case-colliding Hugging Face identity")]
    DuplicateHuggingFaceId,
    #[error("download catalog entry {entry_index} has invalid or overlapping included paths")]
    InvalidIncludedPaths { entry_index: usize },
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
    description: Option<String>,
    capabilities: Option<DownloadCatalogCapabilitiesDocument>,
    quantization_label: Option<String>,
    architecture_summary: Option<String>,
    upstream_license: Option<String>,
    included_paths: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadCatalogCapabilitiesDocument {
    #[serde(default)]
    supports_reasoning: bool,
    #[serde(default)]
    supports_vision: bool,
    #[serde(default)]
    supports_tool_calls: bool,
    context_window: Option<u32>,
    max_output_tokens: Option<u32>,
    #[serde(default)]
    supports_image_generation: bool,
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
            if !is_valid_huggingface_id(&entry_document.huggingface_id) {
                return Err(DownloadCatalogError::InvalidHuggingFaceId { entry_index });
            }
            if !normalized_huggingface_ids
                .insert(entry_document.huggingface_id.to_ascii_lowercase())
            {
                return Err(DownloadCatalogError::DuplicateHuggingFaceId);
            }
            if !is_valid_immutable_revision(&entry_document.revision) {
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
            let description = entry_document
                .description
                .filter(|description| !description.trim().is_empty())
                .map(|description| {
                    if description.len() > MAXIMUM_DESCRIPTION_BYTES
                        || description.chars().any(char::is_control)
                    {
                        return Err(DownloadCatalogError::InvalidDescription { entry_index });
                    }
                    Ok(description)
                })
                .transpose()?;
            let capabilities = entry_document
                .capabilities
                .map(|capabilities_document| {
                    if capabilities_document.supports_reasoning
                        || capabilities_document.supports_vision
                        || capabilities_document.supports_tool_calls
                        || capabilities_document.context_window.is_some()
                        || capabilities_document.max_output_tokens.is_some()
                        || capabilities_document.supports_image_generation
                    {
                        Ok(DownloadCatalogCapabilities {
                            supports_reasoning: capabilities_document.supports_reasoning,
                            supports_vision: capabilities_document.supports_vision,
                            supports_tool_calls: capabilities_document.supports_tool_calls,
                            context_window: capabilities_document.context_window,
                            max_output_tokens: capabilities_document.max_output_tokens,
                            supports_image_generation: capabilities_document
                                .supports_image_generation,
                        })
                    } else {
                        Err(DownloadCatalogError::InvalidCapabilities { entry_index })
                    }
                })
                .transpose()?
                .unwrap_or_default();
            let quantization_label = entry_document
                .quantization_label
                .filter(|label| !label.trim().is_empty())
                .map(|label| {
                    if label.len() > MAXIMUM_QUANTIZATION_LABEL_BYTES
                        || label.chars().any(char::is_control)
                    {
                        return Err(DownloadCatalogError::InvalidQuantizationLabel { entry_index });
                    }
                    Ok(label)
                })
                .transpose()?;
            let architecture_summary = entry_document
                .architecture_summary
                .filter(|summary| !summary.trim().is_empty())
                .map(|summary| {
                    if summary.len() > MAXIMUM_ARCHITECTURE_SUMMARY_BYTES
                        || summary.chars().any(char::is_control)
                    {
                        return Err(DownloadCatalogError::InvalidArchitectureSummary {
                            entry_index,
                        });
                    }
                    Ok(summary)
                })
                .transpose()?;
            let upstream_license = entry_document
                .upstream_license
                .filter(|license| !license.trim().is_empty())
                .map(|license| {
                    if license.len() > MAXIMUM_UPSTREAM_LICENSE_BYTES
                        || license.chars().any(char::is_control)
                    {
                        return Err(DownloadCatalogError::InvalidUpstreamLicense { entry_index });
                    }
                    Ok(license)
                })
                .transpose()?;
            let download_path_selection =
                DownloadPathSelection::try_new(entry_document.included_paths)
                    .map_err(|()| DownloadCatalogError::InvalidIncludedPaths { entry_index })?;
            entries.push(DownloadCatalogEntry {
                huggingface_id: entry_document.huggingface_id,
                revision: entry_document.revision,
                display_name: entry_document.display_name,
                family: entry_document.family,
                approximate_size_bytes: entry_document.approximate_size_bytes,
                description,
                capabilities,
                quantization_label,
                architecture_summary,
                upstream_license,
                download_path_selection,
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

    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    #[must_use]
    pub const fn capabilities(&self) -> &DownloadCatalogCapabilities {
        &self.capabilities
    }

    #[must_use]
    pub fn quantization_label(&self) -> Option<&str> {
        self.quantization_label.as_deref()
    }

    #[must_use]
    pub fn architecture_summary(&self) -> Option<&str> {
        self.architecture_summary.as_deref()
    }

    #[must_use]
    pub fn upstream_license(&self) -> Option<&str> {
        self.upstream_license.as_deref()
    }

    #[must_use]
    pub const fn download_path_selection(&self) -> &DownloadPathSelection {
        &self.download_path_selection
    }
}

pub(crate) fn is_valid_huggingface_id(huggingface_id: &str) -> bool {
    let mut identity_components = huggingface_id.split('/');
    let organization = identity_components.next().unwrap_or_default();
    let model_name = identity_components.next().unwrap_or_default();
    identity_components.next().is_none()
        && is_valid_hugging_face_component(organization)
        && is_valid_hugging_face_component(model_name)
        && !model_name.ends_with(".git")
}

pub(crate) fn is_valid_immutable_revision(revision: &str) -> bool {
    revision.len() == GIT_COMMIT_SHA_HEX_CHARACTER_COUNT
        && revision
            .bytes()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character))
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
