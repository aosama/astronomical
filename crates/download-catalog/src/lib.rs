//! The model download catalog shipped with each release: the entries in
//! `registry/download_catalog.json` and the parser that validates them.
//!
//! It lives apart from the engine so that a change to the catalog alone can be
//! built and tested without the engine's native MLX build; CI's catalog-only
//! lane runs this crate's tests and nothing else.

mod canonical_path;
mod catalog_builds;
mod catalog_capabilities;
mod catalog_classification;
mod catalog_presentation;
mod download_catalog;
mod path_selection;

#[cfg(test)]
mod download_catalog_tests;
#[cfg(test)]
mod publisher_dflash_package_tests;

pub use canonical_path::is_canonical_ascii_path;
pub use download_catalog::{
    DownloadCatalog, DownloadCatalogBuild, DownloadCatalogBuildDifference,
    DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadCatalogError, DownloadCatalogFamily,
    DownloadCatalogRole, is_valid_huggingface_id, is_valid_immutable_revision,
};
pub use path_selection::{DownloadPathSelection, InvalidDownloadPathSelection};
