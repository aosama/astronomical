//! The suggestion that compares builds of one model: a build that keeps more
//! of the original detail.

use crate::catalog_builds::DownloadCatalogBuildDifference;
use crate::download_catalog::DownloadCatalogEntry;

use super::{DecidedSuggestion, MemoryFit, ModelSuggestionKind, SuggestionRules};

impl<'catalog> SuggestionRules<'catalog> {
    /// A build of the best fit that keeps more of the original detail and still
    /// leaves room; the largest such build wins. A build of that model the
    /// person already has that is at least as large (a 6-bit build with a draft
    /// model, say) already keeps that detail, so then nothing is offered.
    pub(super) fn more_detail_suggestion(
        &self,
        best_fit_entry: Option<&'catalog DownloadCatalogEntry>,
    ) -> Option<DecidedSuggestion<'catalog>> {
        let best_fit_entry = best_fit_entry.filter(|entry| self.fits_in_memory(entry))?;
        let largest_entry = self
            .builds_of(best_fit_entry, DownloadCatalogBuildDifference::MoreDetail)
            .filter(|entry| self.fits_with_room(entry))
            .max_by_key(|entry| entry.approximate_memory_bytes())?;
        let has_as_detailed_build = self.family_of(best_fit_entry).any(|entry| {
            self.is_claimed(entry)
                && entry.approximate_memory_bytes() >= largest_entry.approximate_memory_bytes()
        });
        if has_as_detailed_build {
            return None;
        }
        Some(DecidedSuggestion {
            kind: ModelSuggestionKind::MoreDetail,
            offered_entry: largest_entry,
            stronger_than_entry: None,
            reference_entry: Some(best_fit_entry),
            offered_entry_fit: MemoryFit::InMemory,
        })
    }
}
