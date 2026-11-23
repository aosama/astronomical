//! What the engines tell status about a drafter: the resident engine about
//! the one it loaded, the SSD-streaming engine about one it cannot use.
//!
//! Kept apart from loading so every answer is testable without a graphics
//! processor: the resident answers depend only on what loading found (a
//! failed load is also logged), and the streaming answer reads only the
//! library's folders. Reasons are
//! written for the person reading status and never carry a local path. Each
//! starts at the cause, because every surface that shows one already says a
//! drafter is installed and not in use.

use ipc_protocol::{DraftModelGeneration, DraftModelStatus};

use crate::speculative_decoding::dflash_drafter::DrafterGeneration;

use super::engine_draft_runtime::{
    Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftLoadingError,
    Qwen3_5SpeculativeDraftRuntime,
};

/// Reason given when a drafter pairs but the model streams from the SSD.
pub const STREAMING_REASON: &str = "the model is reading its weights from the SSD, and drafting runs only when the whole model is in memory";

/// Reason given when a drafter pairs but this GPU lacks the kernels.
pub const KERNELS_UNAVAILABLE_REASON: &str =
    "this Mac's GPU cannot run the kernels drafting needs; an M5-family GPU can";

fn drafting_status(draft_runtime: &Qwen3_5SpeculativeDraftRuntime) -> DraftModelStatus {
    DraftModelStatus::Drafting {
        generation: reported_generation(draft_runtime.drafter_configuration.generation),
        location: draft_runtime.drafter_location,
        tree_node_budget: u32::try_from(draft_runtime.tree_options.node_budget).unwrap_or(u32::MAX),
    }
}

/// The status for what looking for a drafter found.
#[must_use]
pub fn load_outcome_status(outcome: &Qwen3_5DraftRuntimeLoadOutcome) -> DraftModelStatus {
    match outcome {
        Qwen3_5DraftRuntimeLoadOutcome::Loaded(draft_runtime) => drafting_status(draft_runtime),
        Qwen3_5DraftRuntimeLoadOutcome::NoPairedDrafter => DraftModelStatus::None,
        Qwen3_5DraftRuntimeLoadOutcome::KernelsUnavailable => DraftModelStatus::Unavailable {
            reason: KERNELS_UNAVAILABLE_REASON.to_owned(),
        },
    }
}

/// The drafter the engine keeps and what status says about it, from what
/// looking for one found. A drafter that fails to load costs the speedup,
/// never the model, so a failure is logged and the model decodes classic.
pub fn draft_runtime_and_status(
    load_result: Result<Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftLoadingError>,
) -> (Option<Qwen3_5SpeculativeDraftRuntime>, DraftModelStatus) {
    match load_result {
        Ok(Qwen3_5DraftRuntimeLoadOutcome::Loaded(draft_runtime)) => {
            let draft_model_status = drafting_status(&draft_runtime);
            (Some(*draft_runtime), draft_model_status)
        }
        Ok(load_outcome) => (None, load_outcome_status(&load_outcome)),
        Err(draft_loading_error) => {
            tracing::warn!(
                %draft_loading_error,
                "the installed DFlash drafter could not be loaded; decoding stays classic"
            );
            (None, failed_load_status(&draft_loading_error))
        }
    }
}

/// The status for a drafter that paired but failed to load. The model still
/// serves; it decodes one token at a time.
#[must_use]
pub fn failed_load_status(loading_error: &Qwen3_5SpeculativeDraftLoadingError) -> DraftModelStatus {
    let reason = match loading_error {
        Qwen3_5SpeculativeDraftLoadingError::InsufficientHeadroom { .. } => {
            "the memory limit leaves no room for it beside the model"
        }
        Qwen3_5SpeculativeDraftLoadingError::WeightsUnreadable { .. }
        | Qwen3_5SpeculativeDraftLoadingError::Weights(_) => "its weights could not be read",
        Qwen3_5SpeculativeDraftLoadingError::LibraryUnlistable { .. } => {
            "the model library could not be searched for one"
        }
        Qwen3_5SpeculativeDraftLoadingError::Runtime(_)
        | Qwen3_5SpeculativeDraftLoadingError::Target(_) => "the GPU could not prepare it",
    };
    DraftModelStatus::Unavailable {
        reason: reason.to_owned(),
    }
}

/// The status for a model the SSD-streaming engine loaded, which never
/// drafts: a drafter that would pair is reported as unused, with the reason,
/// so a person knows what they would gain from more memory.
#[must_use]
pub fn streaming_load_status(
    target_model_directory: &std::path::Path,
    target_configuration: &crate::Qwen3_5Config,
) -> DraftModelStatus {
    let pairing = super::drafter_discovery::DrafterPairing {
        target_layer_count: target_configuration.layer_count() as i32,
        target_hidden_size: target_configuration.hidden_size() as i32,
        target_vocabulary_size: target_configuration.vocabulary_size() as i32,
    };
    match super::drafter_discovery::paired_drafter_directory(target_model_directory, &pairing) {
        Ok(Some(_)) => DraftModelStatus::Unavailable {
            reason: STREAMING_REASON.to_owned(),
        },
        Ok(None) => DraftModelStatus::None,
        Err(discovery_error) => failed_load_status(&discovery_error),
    }
}

const fn reported_generation(generation: DrafterGeneration) -> DraftModelGeneration {
    match generation {
        DrafterGeneration::DFlash => DraftModelGeneration::Dflash,
        DrafterGeneration::DFlash2 => DraftModelGeneration::Dflash2,
    }
}

#[cfg(test)]
#[path = "draft_model_status_report_tests.rs"]
mod tests;
