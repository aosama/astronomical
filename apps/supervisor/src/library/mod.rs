//! Release-bundled model catalog and its local Library REST boundary.

mod catalog_endpoint;
mod download_catalog;

pub(crate) use catalog_endpoint::library_catalog_routes;
pub use download_catalog::{
    DownloadCatalog, DownloadCatalogEntry, DownloadCatalogError, DownloadCatalogFamily,
};
