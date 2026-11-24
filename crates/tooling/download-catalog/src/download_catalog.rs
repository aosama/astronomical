//! Strict immutable contract for release-authored model metadata.

use std::collections::BTreeSet;

use serde::Deserialize;
use thiserror::Error;

pub use crate::catalog_builds::{DownloadCatalogBuild, DownloadCatalogBuildDifference};
use crate::catalog_builds::{parse_entry_build, validate_catalog_builds};
pub use crate::catalog_capabilities::DownloadCatalogCapabilities;
use crate::catalog_capabilities::{DownloadCatalogCapabilitiesDocument, parse_entry_capabilities};
pub use crate::catalog_classification::{DownloadCatalogFamily, DownloadCatalogRole};
use crate::catalog_presentation::{CatalogEntryPresentation, parse_entry_presentation};
use crate::catalog_speed::validate_measured_speed;
pub use crate::catalog_speed::{DownloadCatalogMeasuredSpeed, DownloadCatalogTaskSpeeds};
use crate::entry_identity::{SeenEntryIdentities, claim_former_identities};
pub use crate::entry_identity::{is_valid_huggingface_id, is_valid_immutable_revision};
use crate::path_selection::DownloadPathSelection;

// Version 5 removes every name but the model ID: the display name, short name
// and version label gave one model up to four names, and people could not tell
// which one the terminal and the API accept. The model ID is the leaf of the
// Hugging Face identity, the same name discovery gives the downloaded folder.
// Version 5 later gained two optional additions without a version bump: the
// build_of and build_difference pair, which links a build to its lead without
// giving it a second name, and measured_speed, the tokens per second a release
// measured. It later also gained former_huggingface_ids, the names a
// repository had before a rename on Hugging Face, so a copy downloaded under
// an old name stays this entry's copy instead of showing as a second model.
// The catalog is bundled with the binary that reads it, so no reader meets a
// field it does not know.
// Version 4 added the summary; version 3 the memory footprint, streaming floor,
// role, Get started pick, and the experimental flag; its retiring flag was
// removed because a supported model stays in the catalog, and a catalog that
// still declares it is refused. There are no
// downstream catalog consumers, so the schema moves forward without
// compatibility shims.
const DOWNLOAD_CATALOG_SCHEMA_VERSION: u32 = 5;
const MAXIMUM_DOWNLOAD_CATALOG_BYTES: usize = 1_000_000;
const MAXIMUM_DOWNLOAD_CATALOG_ENTRY_COUNT: usize = 1_024;
const MAXIMUM_DESCRIPTION_BYTES: usize = 512;
const MAXIMUM_QUANTIZATION_LABEL_BYTES: usize = 64;
const MAXIMUM_UPSTREAM_LICENSE_BYTES: usize = 128;
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const BUNDLED_DOWNLOAD_CATALOG_JSON: &str = include_str!("../../../../registry/download_catalog.json");

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
    former_huggingface_ids: Vec<String>,
    revision: String,
    family: DownloadCatalogFamily,
    approximate_size_bytes: u64,
    approximate_memory_bytes: u64,
    role: DownloadCatalogRole,
    get_started: bool,
    experimental: bool,
    minimum_memory_bytes: Option<u64>,
    description: Option<String>,
    capabilities: DownloadCatalogCapabilities,
    quantization_label: Option<String>,
    upstream_license: Option<String>,
    presentation: CatalogEntryPresentation,
    build: Option<DownloadCatalogBuild>,
    measured_speed: Option<DownloadCatalogMeasuredSpeed>,
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
    #[error("download catalog entry {entry_index} has an invalid former Hugging Face identity")]
    InvalidFormerHuggingFaceId { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid immutable revision")]
    InvalidRevision { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid description")]
    InvalidDescription { entry_index: usize },
    #[error("download catalog entry {entry_index} has invalid capabilities")]
    InvalidCapabilities { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid quantization label")]
    InvalidQuantizationLabel { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid upstream license")]
    InvalidUpstreamLicense { entry_index: usize },
    #[error("download catalog entry {entry_index} has an invalid summary")]
    InvalidSummary { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} has the same model ID as an earlier entry, ignoring case"
    )]
    DuplicateModelId { entry_index: usize },
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
        "download catalog entry {entry_index} cannot be the get_started pick: it is experimental or a helper"
    )]
    InvalidGetStartedPick { entry_index: usize },
    #[error("download catalog declares more than one get_started pick for role {role}")]
    DuplicateGetStartedPick { role: &'static str },
    #[error("download catalog entry {entry_index} has invalid or overlapping included paths")]
    InvalidIncludedPaths { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} must declare build_of and build_difference together, name another entry, and be neither experimental nor a get_started pick"
    )]
    InvalidBuild { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} names a build_of lead no entry has as its model ID"
    )]
    UnknownBuildLead { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} names a build_of lead that is itself a build, experimental, or listed in another section"
    )]
    InvalidBuildLead { entry_index: usize },
    #[error(
        "download catalog entry {entry_index} has a measured_speed with a rate outside 1 to 1000 tokens a second, an empty or long measured_on, or a version that is not numeric"
    )]
    InvalidMeasuredSpeed { entry_index: usize },
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
    /// Names the repository had before it was renamed on Hugging Face.
    #[serde(default)]
    former_huggingface_ids: Vec<String>,
    revision: String,
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
    minimum_memory_bytes: Option<u64>,
    description: Option<String>,
    capabilities: Option<DownloadCatalogCapabilitiesDocument>,
    quantization_label: Option<String>,
    upstream_license: Option<String>,
    summary: Option<String>,
    included_paths: Option<Vec<String>>,
    /// The model ID of the lead entry this is another build of.
    build_of: Option<String>,
    build_difference: Option<DownloadCatalogBuildDifference>,
    measured_speed: Option<DownloadCatalogMeasuredSpeed>,
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

/// Parses every entry in authored order. The rules that span entries, unique
/// identities and one Get started pick per role, are applied as each entry is
/// read; the build links are checked last, since a lead may follow its builds.
fn parse_catalog_entries(
    entry_documents: Vec<DownloadCatalogEntryDocument>,
) -> Result<Vec<DownloadCatalogEntry>, DownloadCatalogError> {
    let entries = parse_entries_in_authored_order(entry_documents)?;
    validate_catalog_builds(&entries)?;
    Ok(entries)
}

/// Every entry parsed on its own, with the identity and Get started rules
/// that span entries applied as each one is read.
fn parse_entries_in_authored_order(
    entry_documents: Vec<DownloadCatalogEntryDocument>,
) -> Result<Vec<DownloadCatalogEntry>, DownloadCatalogError> {
    let mut seen_identities = SeenEntryIdentities::default();
    let mut get_started_role_names = BTreeSet::new();
    let mut entries = Vec::with_capacity(entry_documents.len());
    for (entry_index, entry_document) in entry_documents.into_iter().enumerate() {
        entries.push(parse_catalog_entry(
            entry_document,
            entry_index,
            &mut seen_identities,
            &mut get_started_role_names,
        )?);
    }
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
    let presentation = parse_entry_presentation(entry_document.summary.as_deref(), entry_index)?;
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
    seen_identities: &mut SeenEntryIdentities,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<DownloadCatalogEntry, DownloadCatalogError> {
    validate_entry_document(
        &entry_document,
        entry_index,
        seen_identities,
        get_started_role_names,
    )?;
    let description = parse_optional_entry_description(&entry_document, entry_index)?;
    let capabilities = parse_entry_capabilities(entry_document.capabilities.as_ref(), entry_index)?;
    let entry_texts = parse_entry_texts(&entry_document, entry_index)?;
    let entry_build_and_files = parse_entry_build_and_files(
        entry_document.included_paths,
        entry_document.build_of.as_deref(),
        entry_document.build_difference,
        entry_document.measured_speed.as_ref(),
        entry_index,
    )?;
    Ok(DownloadCatalogEntry {
        huggingface_id: entry_document.huggingface_id,
        former_huggingface_ids: entry_document.former_huggingface_ids,
        revision: entry_document.revision,
        family: entry_document.family,
        approximate_size_bytes: entry_document.approximate_size_bytes,
        approximate_memory_bytes: entry_document.approximate_memory_bytes,
        role: entry_document.role,
        get_started: entry_document.get_started,
        experimental: entry_document.experimental,
        minimum_memory_bytes: entry_document.minimum_memory_bytes,
        description,
        capabilities,
        quantization_label: entry_texts.quantization_label,
        upstream_license: entry_texts.upstream_license,
        presentation: entry_texts.presentation,
        build: entry_build_and_files.build,
        measured_speed: entry_document.measured_speed,
        download_path_selection: entry_build_and_files.download_path_selection,
    })
}

/// What an entry says about its files and its place among builds: the
/// included paths and the build it belongs to. Its measured speed is checked
/// alongside them but kept on the entry as written.
struct EntryBuildAndFiles {
    download_path_selection: DownloadPathSelection,
    build: Option<DownloadCatalogBuild>,
}

/// Checks the included paths, the build link and the measured speed, in that
/// authored order, so a multi-invalid entry reports the same first failure.
fn parse_entry_build_and_files(
    included_paths: Option<Vec<String>>,
    build_of: Option<&str>,
    build_difference: Option<DownloadCatalogBuildDifference>,
    measured_speed: Option<&DownloadCatalogMeasuredSpeed>,
    entry_index: usize,
) -> Result<EntryBuildAndFiles, DownloadCatalogError> {
    let download_path_selection = DownloadPathSelection::try_new(included_paths)
        .map_err(|_invalid_selection| DownloadCatalogError::InvalidIncludedPaths { entry_index })?;
    let build = parse_entry_build(build_of, build_difference, entry_index)?;
    validate_measured_speed(measured_speed, entry_index)?;
    Ok(EntryBuildAndFiles {
        download_path_selection,
        build,
    })
}

/// Runs the whole-entry checks that need no conversion, in the authored order
/// (identity, size and visibility, memory, Get started pick), so the first
/// failing rule decides the reported error exactly as before they were grouped.
fn validate_entry_document(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    seen_identities: &mut SeenEntryIdentities,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<(), DownloadCatalogError> {
    validate_entry_identity(entry_document, entry_index, seen_identities)?;
    validate_entry_size_and_visibility(entry_document, entry_index)?;
    validate_entry_memory_footprint(entry_document, entry_index)?;
    validate_get_started_pick(entry_document, entry_index, get_started_role_names)
}

/// Guards the entry identity: a Hub-safe repository id, first-seen
/// case-insensitively across the whole catalog, a model ID no earlier entry
/// uses, an immutable full-length commit revision instead of a moving ref,
/// then any former names, held to the same uniqueness as current ones.
/// The model ID is the only name people see, so two organizations publishing
/// the same repository name would show two models no one could tell apart.
fn validate_entry_identity(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    seen_identities: &mut SeenEntryIdentities,
) -> Result<(), DownloadCatalogError> {
    if !is_valid_huggingface_id(&entry_document.huggingface_id) {
        return Err(DownloadCatalogError::InvalidHuggingFaceId { entry_index });
    }
    seen_identities.claim(&entry_document.huggingface_id, entry_index)?;
    if !is_valid_immutable_revision(&entry_document.revision) {
        return Err(DownloadCatalogError::InvalidRevision { entry_index });
    }
    claim_former_identities(
        &entry_document.former_huggingface_ids,
        entry_index,
        seen_identities,
    )
}

/// Guards the JavaScript-safe size claim and the public-visibility release
/// rule; size failures are reported before visibility failures.
fn validate_entry_size_and_visibility(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
) -> Result<(), DownloadCatalogError> {
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
/// a role, so it must be stable (not experimental) and must serve a user-facing
/// role (helpers run behind the scenes). A role can have only one pick, since
/// two picks would make the Library's recommendation ambiguous.
fn validate_get_started_pick(
    entry_document: &DownloadCatalogEntryDocument,
    entry_index: usize,
    get_started_role_names: &mut BTreeSet<&'static str>,
) -> Result<(), DownloadCatalogError> {
    if !entry_document.get_started {
        return Ok(());
    }
    if entry_document.experimental || entry_document.role.serves_other_programs() {
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

    /// Every name the repository has had on Hugging Face, the current one
    /// first. A copy downloaded before a rename keeps the old name in its
    /// provenance and folder, and Hugging Face moves a repository with its
    /// commits, so that copy is still this entry at this revision.
    pub fn huggingface_ids(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.huggingface_id.as_str())
            .chain(self.former_huggingface_ids.iter().map(String::as_str))
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The model's one name, shown everywhere and accepted by the terminal and
    /// the API: the leaf of its Hugging Face identity, which is also the folder
    /// discovery publishes a downloaded copy under.
    #[must_use]
    pub fn model_id(&self) -> &str {
        config::leaf_model_id(&self.huggingface_id)
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

    /// One line saying what the model is for, shown on its row.
    #[must_use]
    pub fn summary(&self) -> Option<&str> {
        self.presentation.summary.as_deref()
    }

    /// The lead entry this is another build of, and what sets it apart, or
    /// `None` for a lead or a model with one build.
    #[must_use]
    pub const fn build(&self) -> Option<&DownloadCatalogBuild> {
        self.build.as_ref()
    }

    /// Reply speeds measured on a named Mac, or `None` when this entry has not
    /// been measured; the Models page then shows no speed rather than a guess.
    #[must_use]
    pub const fn measured_speed(&self) -> Option<&DownloadCatalogMeasuredSpeed> {
        self.measured_speed.as_ref()
    }

    #[must_use]
    pub const fn download_path_selection(&self) -> &DownloadPathSelection {
        &self.download_path_selection
    }
}

#[cfg(test)]
#[path = "download_catalog_parse_tests.rs"]
mod download_catalog_parse_tests;

#[cfg(test)]
#[path = "download_catalog_flag_tests.rs"]
mod download_catalog_flag_tests;
