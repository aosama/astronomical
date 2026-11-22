//! Model catalog: the projection onto installed models and the REST endpoint.
//! The catalog itself and its parser live in `download-catalog`.

mod catalog_copy_match;
mod catalog_endpoint;
mod catalog_projection;
mod catalog_suggestions;

#[cfg(test)]
mod catalog_copy_match_tests;
#[cfg(test)]
mod catalog_load_attribution_tests;

pub(crate) use catalog_endpoint::library_catalog_routes;
pub(crate) use catalog_projection::{
    CatalogEntryProjection, project_catalog_entries, requestable_model_id_from_huggingface_id,
};
pub use download_catalog::{
    DownloadCatalog, DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadCatalogError,
    DownloadCatalogFamily, DownloadCatalogRole,
};
pub(crate) use download_catalog::{is_valid_huggingface_id, is_valid_immutable_revision};
