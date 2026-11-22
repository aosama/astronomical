//! The suggestions that compare builds of one model: a draft build that
//! measured quicker, and a build that keeps more of the original detail.

use crate::catalog_builds::DownloadCatalogBuildDifference;
use crate::download_catalog::DownloadCatalogEntry;

use super::{
    DecidedSuggestion, MemoryFit, ModelSuggestionKind, QUICKER_SPEED_RATIO, SuggestionRules,
};

/// Mean tokens a second for writing an essay across the short and long prompt.
pub(crate) fn essay_rate(entry: &DownloadCatalogEntry) -> Option<f64> {
    let task_speeds = entry.measured_speed()?.tokens_per_second();
    Some(f64::from(task_speeds.essay_short_prompt + task_speeds.essay_long_prompt) / 2.0)
}

/// Mean tokens a second for writing code across the short and long prompt.
pub(crate) fn code_rate(entry: &DownloadCatalogEntry) -> Option<f64> {
    let task_speeds = entry.measured_speed()?.tokens_per_second();
    Some(f64::from(task_speeds.code_short_prompt + task_speeds.code_long_prompt) / 2.0)
}

fn is_clearly_quicker(candidate_rate: Option<f64>, reference_rate: Option<f64>) -> bool {
    match (candidate_rate, reference_rate) {
        (Some(candidate_rate), Some(reference_rate)) => {
            candidate_rate >= reference_rate * QUICKER_SPEED_RATIO
        }
        _ => false,
    }
}

impl<'catalog> SuggestionRules<'catalog> {
    /// The build a quicker one is measured against: the quickest measured
    /// build of the best fit the person already has, else the best fit itself.
    fn speed_reference_entry(
        &self,
        best_fit_entry: &'catalog DownloadCatalogEntry,
    ) -> Option<&'catalog DownloadCatalogEntry> {
        let quickest_claimed_entry = self
            .family_of(best_fit_entry)
            .filter(|entry| self.is_claimed(entry) && entry.measured_speed().is_some())
            .max_by(|left_entry, right_entry| {
                code_rate(left_entry)
                    .partial_cmp(&code_rate(right_entry))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        quickest_claimed_entry.or_else(|| best_fit_entry.measured_speed().map(|_| best_fit_entry))
    }

    /// A draft build of the best fit that measured clearly quicker at code and
    /// fits whole in memory with room: a build that has to stream drafts
    /// nothing yet, so it is never offered for speed. It must also need at least
    /// the memory of the build it replaces, so speed never quietly costs detail
    /// (a plain 4-bit draft build is quicker than a 4/8-bit build partly
    /// because it keeps less).
    pub(super) fn quicker_suggestion(
        &self,
        best_fit_entry: &'catalog DownloadCatalogEntry,
    ) -> Option<DecidedSuggestion<'catalog>> {
        if !self.fits_in_memory(best_fit_entry) {
            return None;
        }
        let reference_entry = self.speed_reference_entry(best_fit_entry)?;
        let quickest_entry = self
            .builds_of(best_fit_entry, DownloadCatalogBuildDifference::DraftModel)
            .filter(|entry| {
                self.fits_with_room(entry)
                    && entry.approximate_memory_bytes()
                        >= reference_entry.approximate_memory_bytes()
                    && !self.is_claimed(entry)
                    && is_clearly_quicker(code_rate(entry), code_rate(reference_entry))
            })
            .max_by(|left_entry, right_entry| {
                code_rate(left_entry)
                    .partial_cmp(&code_rate(right_entry))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        let is_quicker_for_writing =
            is_clearly_quicker(essay_rate(quickest_entry), essay_rate(reference_entry));
        Some(DecidedSuggestion {
            kind: if is_quicker_for_writing {
                ModelSuggestionKind::Quicker
            } else {
                ModelSuggestionKind::QuickerForCode
            },
            offered_entry: quickest_entry,
            stronger_than_entry: None,
            reference_entry: Some(reference_entry),
            offered_entry_fit: MemoryFit::InMemory,
        })
    }

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
