//! Reveals a downloaded catalog model's folder in Finder.
//!
//! The page sends only the catalog identity; the daemon resolves the folder
//! from its own discovery and download state, so the request can never name a
//! path of its choosing.

use std::path::PathBuf;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::Deserialize;

use crate::application::ApplicationState;
use crate::library::{CatalogEntryProjection, project_catalog_entries};
use crate::rest::finder_reveal::reveal_path_in_finder;

pub(crate) fn library_reveal_routes() -> Router<ApplicationState> {
    Router::new().route("/v1/library/reveal", post(reveal_library_model))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryRevealRequest {
    huggingface_id: String,
}

/// Why a reveal was refused before Finder was asked to do anything.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum LibraryRevealRefusal {
    /// The identity is not in this release's catalog.
    UnknownModel,
    /// The model is in the catalog but has no folder on this Mac yet.
    NotOnThisMac,
}

impl LibraryRevealRefusal {
    const fn status_code(&self) -> StatusCode {
        match self {
            Self::UnknownModel => StatusCode::NOT_FOUND,
            Self::NotOnThisMac => StatusCode::CONFLICT,
        }
    }
}

async fn reveal_library_model(
    State(application_state): State<ApplicationState>,
    Json(reveal_request): Json<LibraryRevealRequest>,
) -> StatusCode {
    match resolve_library_model_folder(&application_state, &reveal_request.huggingface_id).await {
        Ok(model_folder) => reveal_path_in_finder(&model_folder).await,
        Err(reveal_refusal) => reveal_refusal.status_code(),
    }
}

/// Joins the catalog with discovery and download state exactly as the catalog
/// endpoint does, so a folder is revealed only for a model the Library shows
/// as on this Mac.
async fn resolve_library_model_folder(
    application_state: &ApplicationState,
    huggingface_id: &str,
) -> Result<PathBuf, LibraryRevealRefusal> {
    let discovered_models = application_state.discovered_models_snapshot();
    let validated_publications = match application_state.library_download_coordinator.as_ref() {
        Some(download_coordinator) => download_coordinator.validated_publications_snapshot().await,
        None => Default::default(),
    };
    let projections = project_catalog_entries(
        &application_state.download_catalog,
        &discovered_models,
        &validated_publications,
        None,
    );
    let matching_projection = projections
        .iter()
        .find(|projection| projection.catalog_entry.huggingface_id() == huggingface_id);
    // The folder is resolved from the catalog's own identity, never the text
    // the request carried, so only a catalog entry's folder can be revealed.
    let published_download_folder = match (
        matching_projection,
        application_state.library_download_coordinator.as_ref(),
    ) {
        (Some(projection), Some(download_coordinator)) => Some(
            download_coordinator.destination_directory(projection.catalog_entry.huggingface_id()),
        ),
        _ => None,
    };
    ready_model_folder(matching_projection, published_download_folder)
}

/// A discovered model reveals the folder discovery found it in; a model that
/// is ready only through a validated download reveals its published folder.
pub(crate) fn ready_model_folder(
    matching_projection: Option<&CatalogEntryProjection<'_>>,
    published_download_folder: Option<PathBuf>,
) -> Result<PathBuf, LibraryRevealRefusal> {
    let projection = matching_projection.ok_or(LibraryRevealRefusal::UnknownModel)?;
    if !projection.ready_on_this_mac {
        return Err(LibraryRevealRefusal::NotOnThisMac);
    }
    match &projection.discovered_model_directory {
        Some(discovered_folder) => Ok(PathBuf::from(discovered_folder)),
        None => published_download_folder.ok_or(LibraryRevealRefusal::NotOnThisMac),
    }
}

#[cfg(test)]
#[path = "reveal_endpoint_tests.rs"]
mod reveal_endpoint_tests;
