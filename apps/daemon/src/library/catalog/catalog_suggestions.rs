//! The suggestions the catalog response carries: the rules in
//! `download_catalog::suggest_models`, fed with what this engine knows about
//! the Mac. The Models page and the chat app's first run both read them from
//! here, so the two can never suggest different models.

use std::collections::BTreeSet;

use download_catalog::{SuggestionInputs, suggest_models};
use serde::Serialize;

use super::CatalogEntryProjection;
use crate::application::ApplicationState;

#[derive(Debug, Serialize)]
pub(super) struct LibrarySuggestionResponse {
    kind: &'static str,
    huggingface_id: String,
    model_id: String,
    tag: &'static str,
    reason: String,
}

/// Suggestions for this Mac, most useful first; empty while the worker has not
/// reported a memory ceiling, so nothing is suggested on a guess.
pub(super) fn library_suggestions(
    application_state: &ApplicationState,
    projections: &[CatalogEntryProjection<'_>],
) -> Vec<LibrarySuggestionResponse> {
    let claimed_huggingface_ids = claimed_huggingface_ids(application_state, projections);
    let suggestions = suggest_models(&SuggestionInputs {
        entries: application_state.download_catalog.entries(),
        claimed_huggingface_ids: &claimed_huggingface_ids,
        serving_ceiling_bytes: serving_ceiling_bytes(application_state),
    });
    suggestions
        .into_iter()
        .map(|suggestion| LibrarySuggestionResponse {
            kind: suggestion.kind.as_str(),
            huggingface_id: suggestion.huggingface_id,
            model_id: suggestion.model_id,
            tag: suggestion.tag,
            reason: suggestion.reason,
        })
        .collect()
}

/// A model the person already has or has asked for: on this Mac, in a
/// download in any state but idle, or waiting in the download list.
/// Suggesting it again would be noise.
fn claimed_huggingface_ids(
    application_state: &ApplicationState,
    projections: &[CatalogEntryProjection<'_>],
) -> BTreeSet<String> {
    let mut claimed_huggingface_ids: BTreeSet<String> = projections
        .iter()
        .filter(|projection| {
            projection.served_on_this_mac
                || projection
                    .download_state
                    .is_some_and(|download_state| download_state != "idle")
        })
        .map(|projection| projection.catalog_entry.huggingface_id().to_owned())
        .collect();
    if let Some(download_coordinator) = application_state.library_download_coordinator.as_ref() {
        claimed_huggingface_ids.extend(
            download_coordinator
                .queue_snapshot()
                .waiting_huggingface_ids()
                .iter()
                .cloned(),
        );
    }
    claimed_huggingface_ids
}

/// The ceiling the engine loads models within: the user's effective limit when
/// the worker has reported it, else the machine's own, else unknown.
fn serving_ceiling_bytes(application_state: &ApplicationState) -> Option<u64> {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    [
        worker_health_snapshot.mlx_memory_ceiling_bytes,
        worker_health_snapshot.machine_mlx_memory_ceiling_bytes,
    ]
    .into_iter()
    .find(|ceiling_bytes| *ceiling_bytes > 0)
}
