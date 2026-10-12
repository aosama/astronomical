//! Model catalog: family definitions, projection, and the REST endpoint.

mod catalog_endpoint;
mod catalog_projection;
mod download_catalog;

#[cfg(test)]
mod download_catalog_k2_horizon_mova_tests;
#[cfg(test)]
mod download_catalog_ornith_6bit_tests;
#[cfg(test)]
mod download_catalog_qwen3_6_tests;
#[cfg(test)]
mod download_catalog_qwen_image_21_tests;
#[cfg(test)]
mod download_catalog_tests;

pub(crate) use catalog_endpoint::library_catalog_routes;
pub(crate) use catalog_projection::{
    CatalogEntryProjection, project_catalog_entries, requestable_model_id_from_huggingface_id,
};
pub use download_catalog::{
    DownloadCatalog, DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadCatalogError,
    DownloadCatalogFamily,
};
pub(crate) use download_catalog::{is_valid_huggingface_id, is_valid_immutable_revision};
