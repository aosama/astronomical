//! Which catalog models to suggest to this Mac, relative to what the person
//! already has.
//!
//! The Models page and the chat app's first run show the same suggestions, so
//! the rules live here once. A suggestion answers "what would be better on this
//! Mac?": the strongest model on the ladder that runs entirely in memory, a
//! quicker draft build of it, a build that keeps more detail, the strongest
//! model that still runs by streaming from the SSD, and a picture model when
//! there is none. The ladder is the catalog's Get started picks for chat and
//! coding, the models the release authors chose to steer people to, ordered by
//! the memory they need. The words people read are made in
//! `model_suggestion_wording`; this module only decides.

use std::collections::BTreeSet;

use crate::catalog_builds::DownloadCatalogBuildDifference;
use crate::catalog_classification::DownloadCatalogRole;
use crate::download_catalog::DownloadCatalogEntry;
pub use crate::model_suggestion_wording::ModelSuggestion;
use crate::model_suggestion_wording::word_suggestion;

// A build is suggested as a step up only when it leaves a tenth of the model
// memory free: one that just squeezes in leaves no room for a long
// conversation, and the page would be suggesting a slowdown.
const ROOMY_FIT_NUMERATOR: u64 = 9;
const ROOMY_FIT_DENOMINATOR: u64 = 10;
// A draft build counts as quicker only when it measured at least this much
// quicker; smaller gaps sit inside the spread between measurement rounds.
pub(crate) const QUICKER_SPEED_RATIO: f64 = 1.1;
const MAXIMUM_SUGGESTION_COUNT: usize = 3;

/// How a model runs within this Mac's memory for models.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MemoryFit {
    InMemory,
    /// Larger than the memory, but its measured streaming floor fits.
    Streams,
    /// Larger than the memory with no streaming floor that fits: it still
    /// runs, with macOS paging the rest, very slowly.
    TooLarge,
}

/// What a suggestion offers; the wording module says it in words.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelSuggestionKind {
    BestFit,
    Stronger,
    Quicker,
    QuickerForCode,
    MoreDetail,
    Streams,
    Pictures,
}

impl ModelSuggestionKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        config::exhaustive_table!(self, {
            Self::BestFit => "best_fit",
            Self::Stronger => "stronger",
            Self::Quicker => "quicker",
            Self::QuickerForCode => "quicker_for_code",
            Self::MoreDetail => "more_detail",
            Self::Streams => "streams",
            Self::Pictures => "pictures",
        })
    }
}

/// One decided suggestion before it is put into words: the entry offered and
/// the entries its reason compares against.
pub(crate) struct DecidedSuggestion<'catalog> {
    pub(crate) kind: ModelSuggestionKind,
    pub(crate) offered_entry: &'catalog DownloadCatalogEntry,
    /// The smaller ladder model the person has, which the offered one is
    /// stronger than.
    pub(crate) stronger_than_entry: Option<&'catalog DownloadCatalogEntry>,
    /// The build the reason compares speed and memory with: the build a draft
    /// build is quicker than, or the best fit a build with more detail extends.
    pub(crate) reference_entry: Option<&'catalog DownloadCatalogEntry>,
    /// How the offered entry runs on this Mac, which the best-fit reason says.
    pub(crate) offered_entry_fit: MemoryFit,
}

/// What the rules read: the release catalog, the Hugging Face identities the
/// person has on this Mac or has asked for (downloading or waiting), and the
/// memory the engine loads models within.
pub struct SuggestionInputs<'catalog> {
    pub entries: &'catalog [DownloadCatalogEntry],
    pub claimed_huggingface_ids: &'catalog BTreeSet<String>,
    pub serving_ceiling_bytes: Option<u64>,
}

/// At most three suggestions, most useful first. With the memory ceiling
/// unknown nothing is suggested rather than a guess.
#[must_use]
pub fn suggest_models(suggestion_inputs: &SuggestionInputs<'_>) -> Vec<ModelSuggestion> {
    let Some(serving_ceiling_bytes) = suggestion_inputs
        .serving_ceiling_bytes
        .filter(|ceiling| *ceiling > 0)
    else {
        return Vec::new();
    };
    let suggestion_rules = SuggestionRules::new(suggestion_inputs, serving_ceiling_bytes);
    suggestion_rules
        .decided_suggestions()
        .into_iter()
        .take(MAXIMUM_SUGGESTION_COUNT)
        .map(|decided_suggestion| word_suggestion(&decided_suggestion))
        .collect()
}

pub(crate) struct SuggestionRules<'catalog> {
    entries: &'catalog [DownloadCatalogEntry],
    claimed_huggingface_ids: &'catalog BTreeSet<String>,
    serving_ceiling_bytes: u64,
    /// Get started chat and coding picks, smallest memory first.
    ladder: Vec<&'catalog DownloadCatalogEntry>,
    /// Lead model IDs the person has any build of.
    claimed_lead_ids: BTreeSet<&'catalog str>,
}

impl<'catalog> SuggestionRules<'catalog> {
    fn new(suggestion_inputs: &SuggestionInputs<'catalog>, serving_ceiling_bytes: u64) -> Self {
        let entries = suggestion_inputs.entries;
        let mut ladder: Vec<&DownloadCatalogEntry> = entries
            .iter()
            .filter(|entry| is_ladder_entry(entry))
            .collect();
        ladder.sort_by_key(|entry| entry.approximate_memory_bytes());
        let claimed_lead_ids = entries
            .iter()
            .filter(|entry| {
                suggestion_inputs
                    .claimed_huggingface_ids
                    .contains(entry.huggingface_id())
            })
            .map(lead_model_id)
            .collect();
        Self {
            entries,
            claimed_huggingface_ids: suggestion_inputs.claimed_huggingface_ids,
            serving_ceiling_bytes,
            ladder,
            claimed_lead_ids,
        }
    }

    fn decided_suggestions(&self) -> Vec<DecidedSuggestion<'catalog>> {
        let best_fit_entry = self.best_fit_entry();
        let mut decided_suggestions = self.best_fit_and_quicker(best_fit_entry);
        decided_suggestions.extend(self.more_detail_suggestion(best_fit_entry));
        decided_suggestions.extend(self.streaming_suggestion(best_fit_entry));
        decided_suggestions.extend(self.pictures_suggestion());
        decided_suggestions
    }

    pub(crate) fn is_claimed(&self, entry: &DownloadCatalogEntry) -> bool {
        self.claimed_huggingface_ids
            .contains(entry.huggingface_id())
    }

    pub(crate) const fn fits_in_memory(&self, entry: &DownloadCatalogEntry) -> bool {
        entry.approximate_memory_bytes() <= self.serving_ceiling_bytes
    }

    pub(crate) fn fits_with_room(&self, entry: &DownloadCatalogEntry) -> bool {
        u128::from(entry.approximate_memory_bytes()) * u128::from(ROOMY_FIT_DENOMINATOR)
            <= u128::from(self.serving_ceiling_bytes) * u128::from(ROOMY_FIT_NUMERATOR)
    }

    pub(crate) fn memory_fit(&self, entry: &DownloadCatalogEntry) -> MemoryFit {
        if self.fits_in_memory(entry) {
            MemoryFit::InMemory
        } else if self.streams(entry) {
            MemoryFit::Streams
        } else {
            MemoryFit::TooLarge
        }
    }

    fn streams(&self, entry: &DownloadCatalogEntry) -> bool {
        entry
            .minimum_memory_bytes()
            .is_some_and(|minimum_memory_bytes| minimum_memory_bytes <= self.serving_ceiling_bytes)
    }

    pub(crate) fn builds_of(
        &self,
        lead_entry: &DownloadCatalogEntry,
        difference: DownloadCatalogBuildDifference,
    ) -> impl Iterator<Item = &'catalog DownloadCatalogEntry> + '_ {
        let lead_id = lead_entry.model_id().to_owned();
        self.entries.iter().filter(move |entry| {
            entry.build().is_some_and(|build| {
                build.lead_model_id() == lead_id && build.difference() == difference
            })
        })
    }

    /// Every entry that is the lead or a build of it, the person's own copies included.
    pub(crate) fn family_of(
        &self,
        lead_entry: &DownloadCatalogEntry,
    ) -> impl Iterator<Item = &'catalog DownloadCatalogEntry> + '_ {
        let lead_id = lead_entry.model_id().to_owned();
        self.entries
            .iter()
            .filter(move |entry| lead_model_id(entry) == lead_id)
    }

    /// The largest ladder model that runs entirely in memory; on a Mac where
    /// none does, the smallest, which streams the least.
    fn best_fit_entry(&self) -> Option<&'catalog DownloadCatalogEntry> {
        self.ladder
            .iter()
            .rev()
            .find(|entry| self.fits_in_memory(entry))
            .or_else(|| self.ladder.first())
            .copied()
    }

    fn best_fit_and_quicker(
        &self,
        best_fit_entry: Option<&'catalog DownloadCatalogEntry>,
    ) -> Vec<DecidedSuggestion<'catalog>> {
        let best_fit_suggestion = best_fit_entry.and_then(|entry| self.best_fit_suggestion(entry));
        let quicker_suggestion = best_fit_entry.and_then(|entry| self.quicker_suggestion(entry));
        merge_quicker_into_best_fit(best_fit_suggestion, quicker_suggestion)
    }

    fn best_fit_suggestion(
        &self,
        best_fit_entry: &'catalog DownloadCatalogEntry,
    ) -> Option<DecidedSuggestion<'catalog>> {
        if self.claimed_lead_ids.contains(best_fit_entry.model_id()) {
            return None;
        }
        let smaller_claimed_entry = self.ladder.iter().copied().find(|entry| {
            self.claimed_lead_ids.contains(entry.model_id())
                && entry.approximate_memory_bytes() < best_fit_entry.approximate_memory_bytes()
        });
        let is_stronger = self.fits_in_memory(best_fit_entry) && smaller_claimed_entry.is_some();
        Some(DecidedSuggestion {
            kind: if is_stronger {
                ModelSuggestionKind::Stronger
            } else {
                ModelSuggestionKind::BestFit
            },
            offered_entry: best_fit_entry,
            stronger_than_entry: if is_stronger {
                smaller_claimed_entry
            } else {
                None
            },
            reference_entry: None,
            offered_entry_fit: self.memory_fit(best_fit_entry),
        })
    }

    /// The largest ladder model, when it does not fit in memory but streams:
    /// much stronger, and the person decides whether slower replies are worth it.
    fn streaming_suggestion(
        &self,
        best_fit_entry: Option<&'catalog DownloadCatalogEntry>,
    ) -> Option<DecidedSuggestion<'catalog>> {
        let largest_entry = *self.ladder.last()?;
        let is_the_best_fit =
            best_fit_entry.is_some_and(|entry| std::ptr::eq(entry, largest_entry));
        if is_the_best_fit
            || self.claimed_lead_ids.contains(largest_entry.model_id())
            || self.fits_in_memory(largest_entry)
            || !self.streams(largest_entry)
        {
            return None;
        }
        Some(DecidedSuggestion {
            kind: ModelSuggestionKind::Streams,
            offered_entry: largest_entry,
            stronger_than_entry: None,
            reference_entry: None,
            offered_entry_fit: MemoryFit::Streams,
        })
    }

    /// Pictures are a different job, so a Mac with no picture model is told one
    /// exists; the Get started picture pick is the one offered.
    fn pictures_suggestion(&self) -> Option<DecidedSuggestion<'catalog>> {
        let is_picture_entry =
            |entry: &&DownloadCatalogEntry| entry.role() == DownloadCatalogRole::ImageGeneration;
        if self
            .entries
            .iter()
            .filter(is_picture_entry)
            .any(|entry| self.is_claimed(entry))
        {
            return None;
        }
        let picture_pick = self
            .entries
            .iter()
            .filter(is_picture_entry)
            .find(|entry| entry.get_started())?;
        Some(DecidedSuggestion {
            kind: ModelSuggestionKind::Pictures,
            offered_entry: picture_pick,
            stronger_than_entry: None,
            reference_entry: None,
            offered_entry_fit: MemoryFit::InMemory,
        })
    }
}

fn is_ladder_entry(entry: &DownloadCatalogEntry) -> bool {
    entry.get_started()
        && matches!(
            entry.role(),
            DownloadCatalogRole::Chat | DownloadCatalogRole::Coding
        )
        && !entry.experimental()
        && !entry.retiring()
}

pub(crate) fn lead_model_id(entry: &DownloadCatalogEntry) -> &str {
    entry
        .build()
        .map_or_else(|| entry.model_id(), |build| build.lead_model_id())
}

/// When a draft build is quicker at everything and the person has not got the
/// best fit yet, the draft build becomes the best fit itself: one card, the
/// quicker build, instead of two that differ only in speed.
fn merge_quicker_into_best_fit<'catalog>(
    best_fit_suggestion: Option<DecidedSuggestion<'catalog>>,
    quicker_suggestion: Option<DecidedSuggestion<'catalog>>,
) -> Vec<DecidedSuggestion<'catalog>> {
    match (best_fit_suggestion, quicker_suggestion) {
        (Some(best_fit), Some(quicker))
            if quicker.kind == ModelSuggestionKind::Quicker
                && quicker
                    .reference_entry
                    .is_some_and(|reference| std::ptr::eq(reference, best_fit.offered_entry)) =>
        {
            vec![DecidedSuggestion {
                kind: best_fit.kind,
                offered_entry: quicker.offered_entry,
                stronger_than_entry: best_fit.stronger_than_entry,
                reference_entry: Some(best_fit.offered_entry),
                offered_entry_fit: MemoryFit::InMemory,
            }]
        }
        (best_fit, quicker) => best_fit.into_iter().chain(quicker).collect(),
    }
}

// The speed and detail rules compare builds of one model; they live beside the
// others as a child module so they share the rules' private state.
#[path = "model_suggestion_builds.rs"]
pub(crate) mod model_suggestion_builds;

#[cfg(test)]
#[path = "model_suggestions_tests.rs"]
mod model_suggestions_tests;
