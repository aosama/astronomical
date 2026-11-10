//! Strict immutable contract for release-authored model metadata.

use std::collections::BTreeSet;

use serde::Deserialize;
use thiserror::Error;

pub use super::catalog_capabilities::DownloadCatalogCapabilities;
use super::catalog_capabilities::{DownloadCatalogCapabilitiesDocument, parse_entry_capabilities};
pub use super::catalog_classification::{DownloadCatalogFamily, DownloadCatalogRole};
use super::catalog_presentation::{
    AuthoredEntryPresentation, CatalogEntryPresentation, parse_entry_presentation,
    validate_unique_model_versions,
};
use crate::library::DownloadPathSelection;

// Version 4 adds the plain-language presentation texts (short name, version
// label, summary) the Models page groups and labels rows with. Version 3 added
// the memory footprint, streaming floor, role, Get started pick, and the
// experimental and retiring flags. There are no downstream catalog consumers,
// so the schema moves forward without compatibility shims.
const DOWNLOAD_CATALOG_SCHEMA_VERSION: u32 = 4;
const MAXIMUM_DOWNLOAD_CATALOG_BYTES: usize = 1_000_000;
const MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT: usize = 1_024;
const MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH: usize = 96;
const MAXIMUM_DISPLAY_NAME_BYTES: usize = 256;
const MAXIMUM_DESCRIPTION_BYTES: usize = 512;
const MAXIMUM_QUANTIZATION_LABEL_BYTES: usize = 64;
const MAXIMUM_UPSTREAM_LICENSE_BYTES: usize = 128;
const GIT_COMMIT_SHA_HEX_CHARACTER_COUNT: usize = 40;
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const BUNDLED_DOWNLOAD_CATALOG_JSON: &str =
    include_str!("../../../../../registry/download_catalog.json");

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
    approximate_memory_bytes: u64,
    role: DownloadCatalogRole,
    get_started: bool,
    experimental: bool,
    retiring: bool,
    minimum_memory_bytes: Option<u64>,
    description: Option<String>,
    capabilities: DownloadCatalogCapabilities,
    quantization_label: Option<String>,
    upstream_license: Option<String>,
    presentation: CatalogEntryPresentation,
    download_path_selection: DownloadPathSelection,
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
    #[error("download catalog entry {entry_index} has an invalid upstream license")]
    InvalidUpstreamLicense { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid short name")]
    InvalidShortName { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} has an invalid version label, or one without a short name"
    )]
    InvalidVersionLabel { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid summary")]
    InvalidSummary { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} repeats the short name and version label of an earlier entry"
    )]
    DuplicateModelVersion { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare approximate_size_bytes between 1 and 9007199254740991"
    )]
    InvalidApproximateSize { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare approximate_memory_bytes between 1 and 9007199254740991"
    )]
    InvalidApproximateMemory { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare minimum_memory_bytes between 1 and its approximate_memory_bytes"
    )]
    InvalidMinimumMemory { entry_index: usize },
    #[error("download catalog entry {entry_index} must declare public: true")]
    ModelNotPublic { entry_index: usize },
    #[error("download catalog contains a duplicate or case-colliding Hugging Face identity")]
    DuplicateHuggingFaceId,
    #[error(
        "download catalog entry {entry_index} cannot be the get_started pick: it is experimental, retiring, or a helper"
    )]
    InvalidGetStartedPick { entry_index: usize },
    #[error("download catalog declares more than one get_started pick for role {role}")]
    DuplicateGetStartedPick { role: &'static str },
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
    approximate_memory_bytes: u64,
    role: DownloadCatalogRole,
    #[serde(rename = "public")]
    is_public: bool,
    #[serde(default)]
    get_started: bool,
    #[serde(default)]
    experimental: bool,
    /// Marks a model that will leave the catalog in a later release; it stays
    /// downloadable and installed copies keep working until then.
    #[serde(default)]
    retiring: bool,
    minimum_memory_bytes: Option<u64>,
    description: Option<String>,
    capabilities: Option<DownloadCatalogCapabilitiesDocument>,
    quantization_label: Option<String>,
    upstream_license: Option<String>,
    short_name: Option<String>,
    version_label: Option<String>,
    summary: Option<String>,
    included_paths: Option<Vec<String>>,
}

impl DownloadCatalog {
    /// Parses and validates one complete catalog document.
    pub fn parse_json(catalog_json: &str) -> Result<Self, DownloadCatalogError> {
        let catalog_document = parse_and_shape_check_document(catalog_json)?;
        let schema_version = catalog_document.schema_version;
        let entries = parse_catalog_entries(catalog_document.entries)?;
        Ok(Self {
            schema_version,
            entries,
        })
    }

    /// Validates the catalog embedded into the current daemon binary.
    pub fn load_bundled() -> Result<Self, DownloadCatalogError> {
        Self::parse_json(BUNDLED_DOWNLOAD_CATALOG_JSON)
    }

    /// Supplies a validated empty catalog to test-focused application builders.
    #[must_use]
    pub const fn empty() -> Self {
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

/// Enforces the document-level startup resource bounds before any per-entry
/// work: byte size, JSON syntax, schema version, and the entry-count ceiling.
fn parse_and_shape_check_document(
    catalog_json: &str,
) -> Result<DownloadCatalogDocument, DownloadCatalogError> {
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
    Ok(catalog_document)
}

/// Parses every entry in authored order, then applies the one rule that spans
/// entries: no two builds may share a short name and a version label.
fn parse_catalog_entries(
    entry_documents: Vec<DownloadCatalogEntryDocument>,
) -> Result<Vec<DownloadCatalogEntry>, DownloadCatalogError> {
    let mut normalized_huggingface_ids = BTreeSet::new();
    let mut get_started_role_names = BTreeSet::new();
    let mut entries = Vec::with_capacity(entry_documents.len());
    for (entry_index, entry_document) in entry_documents.into_iter().enumerate() {
        entries.push(parse_catalog_entry(
            entry_document,
            entry_index,
            &mut normalized_huggingface_ids,
            &mut get_started_role_names,
        )?);
    }
    validate_unique_model_versions(&entries)?;
    Ok(entries)
}

/// The labels of one entry that are checked after its capabilities, in the
/// authored order: quantization label, upstream license, then the
/// presentation texts. The order decides which failure a multi-invalid entry
/// reports.
struct EntryTexts {
    quantization_label: Option<String>,
    upstream_license: Option<String>,
    presentation: CatalogEntryPresentation,
}

fn parse_entry_texts(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<EntryTexts, DownloadCatalogError> {
    let quantization_label = parse_optional_entry_quantization_label(entry_document, entry_index)?;
    let upstream_license = parse_optional_entry_upstream_license(entry_document, entry_index)?;
    let presentation = parse_entry_presentation(
        &AuthoredEntryPresentation {
            short_name: entry_document.short_name.as_deref(),
            version_label: entry_document.version_label.as_deref(),
            summary: entry_document.summary.as_deref(),
        },
        entry_index,
    )?;
    Ok(EntryTexts {
        quantization_label,
        upstream_license,
        presentation,
    })
}

/// Validates one entry in the authored check order — identity, display
/// metadata, description, capabilities, quantization label, upstream license,
/// presentation texts, then included paths — and converts it into the public entry type. The
/// order decides which failure a multi-invalid entry reports.
fn parse_catalog_entry(
    entry_document: DownloadCatalogEntryDocument,
    entry_index: usize,
    normalized_huggingface_ids: &mut BTreeSet<String>,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<DownloadCatalogEntry, DownloadCatalogError> {
    validate_entry_document(
        &entry_document,
        entry_index,
        normalized_huggingface_ids,
        get_started_role_names,
    )?;
    let description = parse_optional_entry_description(&entry_document, entry_index)?;
    let capabilities = parse_entry_capabilities(entry_document.capabilities.as_ref(), entry_index)?;
    let entry_texts = parse_entry_texts(&entry_document, entry_index)?;
    let download_path_selection = DownloadPathSelection::try_new(entry_document.included_paths)
        .map_err(|()| DownloadCatalogError::InvalidIncludedPaths { entry_index })?;
    Ok(DownloadCatalogEntry {
        huggingface_id: entry_document.huggingface_id,
        revision: entry_document.revision,
        display_name: entry_document.display_name,
        family: entry_document.family,
        approximate_size_bytes: entry_document.approximate_size_bytes,
        approximate_memory_bytes: entry_document.approximate_memory_bytes,
        role: entry_document.role,
        get_started: entry_document.get_started,
        experimental: entry_document.experimental,
        retiring: entry_document.retiring,
        minimum_memory_bytes: entry_document.minimum_memory_bytes,
        description,
        capabilities,
        quantization_label: entry_texts.quantization_label,
        upstream_license: entry_texts.upstream_license,
        presentation: entry_texts.presentation,
        download_path_selection,
    })
}

/// Runs the whole-entry checks that need no conversion, in the authored order
/// (identity, display metadata, memory, Get started pick), so the first failing
/// rule decides the reported error exactly as before they were grouped.
fn validate_entry_document(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    normalized_huggingface_ids: &mut BTreeSet<String>,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<(), DownloadCatalogError> {
    validate_entry_identity(entry_document, entry_index, normalized_huggingface_ids)?;
    validate_entry_display_metadata(entry_document, entry_index)?;
    validate_entry_memory_footprint(entry_document, entry_index)?;
    validate_get_started_pick(entry_document, entry_index, get_started_role_names)
}

/// Guards the entry identity: a Hub-safe repository id, first-seen
/// case-insensitively across the whole catalog, plus an immutable full-length
/// commit revision instead of a moving ref.
fn validate_entry_identity(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    normalized_huggingface_ids: &mut BTreeSet<String>,
) -> Result<(), DownloadCatalogError> {
    if !is_valid_huggingface_id(&entry_document.huggingface_id) {
        return Err(DownloadCatalogError::InvalidHuggingFaceId { entry_index });
    }
    if !normalized_huggingface_ids.insert(entry_document.huggingface_id.to_ascii_lowercase()) {
        return Err(DownloadCatalogError::DuplicateHuggingFaceId);
    }
    if !is_valid_immutable_revision(&entry_document.revision) {
        return Err(DownloadCatalogError::InvalidRevision { entry_index });
    }
    Ok(())
}

/// Guards the human-facing display name, the JavaScript-safe size claim, and
/// the public-visibility release rule; display-name failures are reported
/// before size and visibility failures.
fn validate_entry_display_metadata(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<(), DownloadCatalogError> {
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
    Ok(())
}

/// Memory is bounded by the same JavaScript-safe integer rule as the download
/// size, so the Library can render both figures exactly. Zero would claim a
/// model needs no memory, which the admission check cannot honor. The optional
/// streaming floor must sit inside that footprint.
fn validate_entry_memory_footprint(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<(), DownloadCatalogError> {
    if entry_document.approximate_memory_bytes == 0
        || entry_document.approximate_memory_bytes > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
    {
        return Err(DownloadCatalogError::InvalidApproximateMemory { entry_index });
    }
    // A streaming floor above the fully loaded footprint would make the Library
    // promise streaming exactly where the model already fits whole.
    if let Some(minimum_memory_bytes) = entry_document.minimum_memory_bytes
        && (minimum_memory_bytes == 0
            || minimum_memory_bytes > entry_document.approximate_memory_bytes)
    {
        return Err(DownloadCatalogError::InvalidMinimumMemory { entry_index });
    }
    Ok(())
}

/// The Get started pick is the single model steered to a first-time user for
/// a role, so it must be stable (not experimental), staying in the catalog (not
/// retiring), and must serve a user-facing role (helpers run behind the scenes). A role can have only one pick, since
/// two picks would make the Library's recommendation ambiguous.
fn validate_get_started_pick(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<(), DownloadCatalogError> {
    if !entry_document.get_started {
        return Ok(());
    }
    if entry_document.experimental
        || entry_document.retiring
        || entry_document.role.serves_other_programs()
    {
        return Err(DownloadCatalogError::InvalidGetStartedPick { entry_index });
    }
    let role_name = entry_document.role.as_str();
    if !get_started_role_names.insert(role_name) {
        return Err(DownloadCatalogError::DuplicateGetStartedPick { role: role_name });
    }
    Ok(())
}

/// Whitespace-only or absent descriptions stay absent; a present description
/// must fit the display bound and carry no control characters.
fn parse_optional_entry_description(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<Option<String>, DownloadCatalogError> {
    let Some(description) = entry_document.description.as_deref() else {
        return Ok(None);
    };
    if description.trim().is_empty() {
        return Ok(None);
    }
    if description.len() > MAXIMUM_DESCRIPTION_BYTES || description.chars().any(char::is_control) {
        return Err(DownloadCatalogError::InvalidDescription { entry_index });
    }
    Ok(Some(description.to_owned()))
}

/// Whitespace-only or absent quantization labels stay absent; a present label
/// must fit its bound and carry no control characters.
fn parse_optional_entry_quantization_label(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<Option<String>, DownloadCatalogError> {
    let Some(quantization_label) = entry_document.quantization_label.as_deref() else {
        return Ok(None);
    };
    if quantization_label.trim().is_empty() {
        return Ok(None);
    }
    if quantization_label.len() > MAXIMUM_QUANTIZATION_LABEL_BYTES
        || quantization_label.chars().any(char::is_control)
    {
        return Err(DownloadCatalogError::InvalidQuantizationLabel { entry_index });
    }
    Ok(Some(quantization_label.to_owned()))
}

/// Whitespace-only or absent upstream licenses stay absent; a present license
/// must fit its bound and carry no control characters.
fn parse_optional_entry_upstream_license(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<Option<String>, DownloadCatalogError> {
    let Some(upstream_license) = entry_document.upstream_license.as_deref() else {
        return Ok(None);
    };
    if upstream_license.trim().is_empty() {
        return Ok(None);
    }
    if upstream_license.len() > MAXIMUM_UPSTREAM_LICENSE_BYTES
        || upstream_license.chars().any(char::is_control)
    {
        return Err(DownloadCatalogError::InvalidUpstreamLicense { entry_index });
    }
    Ok(Some(upstream_license.to_owned()))
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
    pub const fn approximate_memory_bytes(&self) -> u64 {
        self.approximate_memory_bytes
    }

    #[must_use]
    pub const fn role(&self) -> DownloadCatalogRole {
        self.role
    }

    #[must_use]
    pub const fn get_started(&self) -> bool {
        self.get_started
    }

    #[must_use]
    pub const fn experimental(&self) -> bool {
        self.experimental
    }

    /// Whether the model will leave the catalog in a later release.
    #[must_use]
    pub const fn retiring(&self) -> bool {
        self.retiring
    }

    /// The smallest memory ceiling the daemon measured this model running at by
    /// streaming weights from the SSD, or `None` when no streaming floor was
    /// measured and the model must fit in memory whole.
    #[must_use]
    pub const fn minimum_memory_bytes(&self) -> Option<u64> {
        self.minimum_memory_bytes
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
    pub fn upstream_license(&self) -> Option<&str> {
        self.upstream_license.as_deref()
    }

    /// The name people read on the Models page; builds of one model share it.
    #[must_use]
    pub fn short_name(&self) -> Option<&str> {
        self.presentation.short_name.as_deref()
    }

    /// Tells this build apart from other builds sharing its short name.
    #[must_use]
    pub fn version_label(&self) -> Option<&str> {
        self.presentation.version_label.as_deref()
    }

    /// One line saying what the model is for, shown on its row.
    #[must_use]
    pub fn summary(&self) -> Option<&str> {
        self.presentation.summary.as_deref()
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
    astronomical_lowercase_hex::is_lowercase_hex_of_length(
        revision,
        GIT_COMMIT_SHA_HEX_CHARACTER_COUNT,
    )
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

#[cfg(test)]
#[path = "download_catalog_parse_tests.rs"]
mod download_catalog_parse_tests;

#[cfg(test)]
#[path = "download_catalog_flag_tests.rs"]
mod download_catalog_flag_tests;
