//! Plain-language presentation metadata of a catalog entry.
//!
//! The Models page shows one row per model rather than one card per build, so
//! each entry may carry the short name people read ("Ornith 9B"), the version
//! label that tells builds of that model apart ("4-bit", "6-bit"), and a
//! one-line summary. The full `display_name` stays the technical identity.

use std::collections::BTreeSet;

use super::download_catalog::{DownloadCatalogEntry, DownloadCatalogError};

pub(super) const MAXIMUM_SHORT_NAME_BYTES: usize = 64;
pub(super) const MAXIMUM_VERSION_LABEL_BYTES: usize = 24;
// One line at the row's reading size; a longer summary wraps and the list stops
// being scannable, which is the reason the summary exists beside the description.
pub(super) const MAXIMUM_SUMMARY_BYTES: usize = 120;

/// The three optional presentation texts of one entry, already validated.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CatalogEntryPresentation {
    pub short_name: Option<String>,
    pub version_label: Option<String>,
    pub summary: Option<String>,
}

/// The authored, unvalidated presentation texts of one entry.
pub(super) struct AuthoredEntryPresentation<'a> {
    pub short_name: Option<&'a str>,
    pub version_label: Option<&'a str>,
    pub summary: Option<&'a str>,
}

/// Validates the presentation texts in authored order: short name, version
/// label, then summary. A version label without a short name is rejected
/// because the label only has meaning beside the model it is a version of.
pub(super) fn parse_entry_presentation(
    authored_presentation: &AuthoredEntryPresentation<'_>,
    entry_index: usize,
) -> Result<CatalogEntryPresentation, DownloadCatalogError> {
    let short_name = parse_bounded_text(authored_presentation.short_name, MAXIMUM_SHORT_NAME_BYTES)
        .ok_or(DownloadCatalogError::InvalidShortName { entry_index })?;
    let version_label = parse_bounded_text(
        authored_presentation.version_label,
        MAXIMUM_VERSION_LABEL_BYTES,
    )
    .ok_or(DownloadCatalogError::InvalidVersionLabel { entry_index })?;
    if version_label.is_some() && short_name.is_none() {
        return Err(DownloadCatalogError::InvalidVersionLabel { entry_index });
    }
    let summary = parse_bounded_text(authored_presentation.summary, MAXIMUM_SUMMARY_BYTES)
        .ok_or(DownloadCatalogError::InvalidSummary { entry_index })?;
    Ok(CatalogEntryPresentation {
        short_name,
        version_label,
        summary,
    })
}

/// Two builds sharing a short name are told apart on the Models page only by
/// their version label, so a repeated pair would offer two choices nobody can
/// tell apart. Names and labels compare case-insensitively for the same reason.
pub(super) fn validate_unique_model_versions(
    catalog_entries: &[DownloadCatalogEntry],
) -> Result<(), DownloadCatalogError> {
    let mut declared_model_versions = BTreeSet::new();
    for (entry_index, catalog_entry) in catalog_entries.iter().enumerate() {
        let Some(short_name) = catalog_entry.short_name() else {
            continue;
        };
        let version_label = catalog_entry.version_label().unwrap_or_default();
        let model_version = (short_name.to_lowercase(), version_label.to_lowercase());
        if !declared_model_versions.insert(model_version) {
            return Err(DownloadCatalogError::DuplicateModelVersion { entry_index });
        }
    }
    Ok(())
}

/// `Some(None)` for absent or whitespace-only text, `Some(Some(text))` for a
/// valid one, and `None` when the text exceeds its bound or carries control
/// characters.
fn parse_bounded_text(authored_text: Option<&str>, maximum_bytes: usize) -> Option<Option<String>> {
    let Some(authored_text) = authored_text else {
        return Some(None);
    };
    if authored_text.trim().is_empty() {
        return Some(None);
    }
    if authored_text.len() > maximum_bytes || authored_text.chars().any(char::is_control) {
        return None;
    }
    Some(Some(authored_text.to_owned()))
}

#[cfg(test)]
#[path = "catalog_presentation_tests.rs"]
mod catalog_presentation_tests;
