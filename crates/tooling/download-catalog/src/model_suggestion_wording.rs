//! The words a suggestion is shown with: a short tag and one or two sentences
//! of reason, naming every model by its model ID and every figure in the
//! 2^30-byte gigabytes the menu bar uses. The rules in `model_suggestions`
//! decide; this module only words, so a change of wording never moves a rule.

use crate::download_catalog::DownloadCatalogEntry;
use crate::model_suggestions::model_suggestion_builds::{code_rate, essay_rate};
use crate::model_suggestions::{DecidedSuggestion, MemoryFit, ModelSuggestionKind};

const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;
// A change smaller than this is inside the measurement spread and is not worth
// a sentence.
const SLOWER_PERCENT_WORTH_SAYING: i64 = -3;

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
        ModelSuggestionKind::Quicker => "Quicker",
        ModelSuggestionKind::QuickerForCode => "Quicker for code",
        ModelSuggestionKind::MoreDetail => "Closer to the original",
        ModelSuggestionKind::Streams => "The strongest that runs here",
        ModelSuggestionKind::Pictures => "Make pictures",
    })
}

fn suggestion_reason(decided_suggestion: &DecidedSuggestion<'_>) -> String {
    let offered_entry = decided_suggestion.offered_entry;
    let reference_entry = decided_suggestion.reference_entry;
    match decided_suggestion.kind {
        ModelSuggestionKind::BestFit | ModelSuggestionKind::Stronger => best_fit_reason(decided_suggestion),
        ModelSuggestionKind::Quicker => reference_entry.map_or_else(String::new, |reference| quicker_reason(offered_entry, reference)),
        ModelSuggestionKind::QuickerForCode => {
            reference_entry.map_or_else(String::new, |reference| quicker_for_code_reason(offered_entry, reference))
        }
        ModelSuggestionKind::MoreDetail => reference_entry.map_or_else(String::new, |reference| more_detail_reason(offered_entry, reference)),
        ModelSuggestionKind::Streams => {
            "Much stronger, but part of it is read from the SSD while it answers, so replies are slower.".to_owned()
        }
        ModelSuggestionKind::Pictures => {
            "Turns a written description into a picture. You have no picture model yet.".to_owned()
        }
    }
}

/// The best fit, or a stronger model than the one the person has; when a
/// quicker draft build took its place, the reason adds what the draft buys.
fn best_fit_reason(decided_suggestion: &DecidedSuggestion<'_>) -> String {
    let offered_entry = decided_suggestion.offered_entry;
    let opening = match (decided_suggestion.stronger_than_entry, decided_suggestion.offered_entry_fit) {
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
    };
    match decided_suggestion.reference_entry {
        Some(replaced_entry) => format!(
            "{opening} Its draft model makes replies about {}% quicker when writing and {}% quicker with code.",
            percent_change(essay_rate(offered_entry), essay_rate(replaced_entry)),
            percent_change(code_rate(offered_entry), code_rate(replaced_entry)),
        ),
        None => opening,
    }
}

fn quicker_reason(
    offered_entry: &DownloadCatalogEntry,
    reference_entry: &DownloadCatalogEntry,
) -> String {
    format!(
        "Replies about {}% quicker when writing and {}% quicker with code than {}, because a draft model is built in.",
        percent_change(essay_rate(offered_entry), essay_rate(reference_entry)),
        percent_change(code_rate(offered_entry), code_rate(reference_entry)),
        reference_entry.model_id(),
    )
}

fn quicker_for_code_reason(
    offered_entry: &DownloadCatalogEntry,
    reference_entry: &DownloadCatalogEntry,
) -> String {
    let essay_change = percent_change(essay_rate(offered_entry), essay_rate(reference_entry));
    let essay_text = if essay_change <= SLOWER_PERCENT_WORTH_SAYING {
        format!(" Writing an essay is about {}% slower.", -essay_change)
    } else {
        String::new()
    };
    format!(
        "Writes code about {}% quicker than {}, because a draft model is built in.{essay_text}",
        percent_change(code_rate(offered_entry), code_rate(reference_entry)),
        reference_entry.model_id(),
    )
}

fn more_detail_reason(
    offered_entry: &DownloadCatalogEntry,
    base_entry: &DownloadCatalogEntry,
) -> String {
    let extra_memory_bytes = offered_entry
        .approximate_memory_bytes()
        .saturating_sub(base_entry.approximate_memory_bytes());
    let speed_change = percent_change(essay_rate(offered_entry), essay_rate(base_entry));
    let speed_text = if speed_change <= SLOWER_PERCENT_WORTH_SAYING {
        format!(", and replies about {}% slower", -speed_change)
    } else {
        String::new()
    };
    format!(
        "The same model as {} with more of its original detail kept. It needs {} more memory{speed_text}.",
        base_entry.model_id(),
        gigabytes_text(extra_memory_bytes),
    )
}

/// Whole-percent change from the reference rate; zero when either is unmeasured.
fn percent_change(new_rate: Option<f64>, reference_rate: Option<f64>) -> i64 {
    match (new_rate, reference_rate) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a percentage of two measured rates is far inside i64"
        )]
        (Some(new_rate), Some(reference_rate)) if reference_rate > 0.0 => {
            ((new_rate / reference_rate - 1.0) * 100.0).round() as i64
        }
        _ => 0,
    }
}

/// "5.3 GB": one decimal in 2^30-byte gigabytes, matching the Models page.
#[expect(
    clippy::cast_precision_loss,
    reason = "model sizes are far below 2^52 bytes"
)]
fn gigabytes_text(byte_count: u64) -> String {
    format!("{:.1} GB", byte_count as f64 / BYTES_PER_GIGABYTE)
}
