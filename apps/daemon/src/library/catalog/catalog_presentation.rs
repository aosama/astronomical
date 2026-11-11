//! Plain-language presentation metadata of a catalog entry.
//!
//! A model has one name, its model ID, so the only presentation text an entry
//! carries beside its description is a one-line summary saying what it is for.

use super::download_catalog::DownloadCatalogError;

// One line at the row's reading size; a longer summary wraps and the list stops
// being scannable, which is the reason the summary exists beside the description.
pub(super) const MAXIMUM_SUMMARY_BYTES: usize = 120;

/// The optional presentation text of one entry, already validated.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CatalogEntryPresentation {
    pub summary: Option<String>,
}

/// Validates the authored summary of one entry.
pub(super) fn parse_entry_presentation(
    authored_summary: Option<&str>,
    entry_index: usize,
) -> Result<CatalogEntryPresentation, DownloadCatalogError> {
    let summary = parse_bounded_text(authored_summary, MAXIMUM_SUMMARY_BYTES)
        .ok_or(DownloadCatalogError::InvalidSummary { entry_index })?;
    Ok(CatalogEntryPresentation { summary })
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
