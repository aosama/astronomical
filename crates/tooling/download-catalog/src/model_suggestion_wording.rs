//! The words a suggestion is shown with: a short tag and one or two sentences
//! of reason, naming every model by its model ID and every figure in the
//! 2^30-byte gigabytes the menu bar uses. The rules in `model_suggestions`
//! decide; this module only words, so a change of wording never moves a rule.

use crate::download_catalog::DownloadCatalogEntry;
use crate::model_suggestions::{DecidedSuggestion, MemoryFit, ModelSuggestionKind};

const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

/// One suggestion as people read it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSuggestion {
    pub kind: ModelSuggestionKind,
    pub huggingface_id: String,
    pub model_id: String,
    pub tag: &'static str,
    pub reason: String,
}

pub(crate) fn word_suggestion(decided_suggestion: &DecidedSuggestion<'_>) -> ModelSuggestion {
    let offered_entry = decided_suggestion.offered_entry;
    ModelSuggestion {
        kind: decided_suggestion.kind,
        huggingface_id: offered_entry.huggingface_id().to_owned(),
        model_id: offered_entry.model_id().to_owned(),
        tag: suggestion_tag(decided_suggestion.kind),
        reason: suggestion_reason(decided_suggestion),
    }
}

const fn suggestion_tag(kind: ModelSuggestionKind) -> &'static str {
    config::exhaustive_table!(kind, {
        ModelSuggestionKind::BestFit => "Best fit for this Mac",
        ModelSuggestionKind::Stronger => "A stronger model",
        ModelSuggestionKind::MoreDetail => "Closer to the original",
        ModelSuggestionKind::Streams => "The strongest that runs here",
        ModelSuggestionKind::Pictures => "Make pictures",
    })
}

fn suggestion_reason(decided_suggestion: &DecidedSuggestion<'_>) -> String {
    let reference_entry = decided_suggestion.reference_entry;
    match decided_suggestion.kind {
        ModelSuggestionKind::BestFit | ModelSuggestionKind::Stronger => {
            best_fit_reason(decided_suggestion)
        }
        ModelSuggestionKind::MoreDetail => reference_entry.map_or_else(String::new, |reference| {
            more_detail_reason(decided_suggestion.offered_entry, reference)
        }),
        ModelSuggestionKind::Streams => {
            "Much stronger, but part of it is read from the SSD while it answers, so replies are slower.".to_owned()
        }
        ModelSuggestionKind::Pictures => {
            "Turns a written description into a picture. You have no picture model yet.".to_owned()
        }
    }
}

/// The best fit, or a stronger model than the one the person has.
fn best_fit_reason(decided_suggestion: &DecidedSuggestion<'_>) -> String {
    match (
        decided_suggestion.stronger_than_entry,
        decided_suggestion.offered_entry_fit,
    ) {
        (Some(smaller_entry), _) => format!(
            "Better at hard questions and code than {}, and it still runs entirely in this Mac's memory.",
            smaller_entry.model_id()
        ),
        (None, MemoryFit::InMemory) => "The strongest suggested model that runs entirely in this Mac's memory.".to_owned(),
        (None, MemoryFit::Streams) => {
            "The lightest of the suggested models. It reads part of itself from the SSD while it answers, so replies are slower.".to_owned()
        }
        (None, MemoryFit::TooLarge) => {
            "The lightest of the suggested models. It needs more memory than this Mac gives to models, so replies are very slow.".to_owned()
        }
    }
}

fn more_detail_reason(
    offered_entry: &DownloadCatalogEntry,
    base_entry: &DownloadCatalogEntry,
) -> String {
    let extra_memory_bytes = offered_entry
        .approximate_memory_bytes()
        .saturating_sub(base_entry.approximate_memory_bytes());
    format!(
        "The same model as {} with more of its original detail kept. It needs {} more memory.",
        base_entry.model_id(),
        gigabytes_text(extra_memory_bytes),
    )
}

/// "5.3 GB": one decimal in 2^30-byte gigabytes, matching the Models page.
#[expect(
    clippy::cast_precision_loss,
    reason = "model sizes are far below 2^52 bytes"
)]
fn gigabytes_text(byte_count: u64) -> String {
    format!("{:.1} GB", byte_count as f64 / BYTES_PER_GIGABYTE)
}
