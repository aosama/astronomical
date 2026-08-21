//! Read-only REST projection of the validated release catalog.

use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::application::ApplicationState;

pub(crate) fn library_catalog_routes() -> Router<ApplicationState> {
    Router::new().route("/v1/library/catalog", get(get_library_catalog))
}

async fn get_library_catalog(
    State(application_state): State<ApplicationState>,
) -> Json<LibraryCatalogResponse> {
    Json(LibraryCatalogResponse::from_catalog(
        &application_state.download_catalog,
    ))
}

#[derive(Debug, Serialize)]
struct LibraryCatalogResponse {
    schema_version: u32,
    entries: Vec<LibraryCatalogEntryResponse>,
}

#[derive(Debug, Serialize)]
struct LibraryCatalogEntryResponse {
    huggingface_id: String,
    revision: String,
    display_name: String,
    family: &'static str,
    approximate_size_bytes: u64,
    public: bool,
}

impl LibraryCatalogResponse {
    fn from_catalog(download_catalog: &super::DownloadCatalog) -> Self {
        Self {
            schema_version: download_catalog.schema_version(),
            entries: download_catalog
                .entries()
                .iter()
                .map(LibraryCatalogEntryResponse::from_entry)
                .collect(),
        }
    }
}

impl LibraryCatalogEntryResponse {
    fn from_entry(catalog_entry: &super::DownloadCatalogEntry) -> Self {
        Self {
            huggingface_id: catalog_entry.huggingface_id().to_owned(),
            revision: catalog_entry.revision().to_owned(),
            display_name: catalog_entry.display_name().to_owned(),
            family: catalog_entry.family().as_str(),
            approximate_size_bytes: catalog_entry.approximate_size_bytes(),
            public: true,
        }
    }
}
