//! Shared projection of the release catalog joined with local readiness state.
//!
//! Both the REST catalog endpoint and the daemon IPC `Catalog` request answer
//! the same question — "is this catalog entry installed on this Mac, and under
//! which requestable ID?" — so the join logic lives here once.

use std::collections::BTreeSet;

use config::DiscoveredModel;

use super::catalog_copy_match::{find_discovered_copy, find_exact_copy};

use crate::library::{
    DownloadCatalog, DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadJob,
};

/// Readiness state of one catalog entry, computed once per request.
pub(crate) struct CatalogEntryProjection<'a> {
    pub catalog_entry: &'a DownloadCatalogEntry,
    pub ready_on_this_mac: bool,
    /// The engine serves a copy of this model, at the catalog's revision or
    /// another (see `find_discovered_copy`). On this Mac lists such a copy
    /// even before Get has verified and adopted it.
    pub served_on_this_mac: bool,
    pub discovered_model_directory: Option<String>,
    pub requestable_model_id: Option<String>,
    pub download_state: Option<&'static str>,
    pub capabilities: DownloadCatalogCapabilities,
}

/// Joins every catalog entry with discovery and publication state.
///
/// An entry is ready when a copy is discovered at the catalog's revision with
/// its provenance, or when a validated download publication exists for its
/// Hugging Face identity. It is served when it is ready or the engine finds
/// any other copy of the same model (see `find_discovered_copy`).
pub(crate) fn project_catalog_entries<'a>(
    download_catalog: &'a DownloadCatalog,
    discovered_models: &[DiscoveredModel],
    validated_publications: &BTreeSet<String>,
    current_job: Option<&DownloadJob>,
) -> Vec<CatalogEntryProjection<'a>> {
    let mut projections = Vec::with_capacity(download_catalog.entry_count());
    for catalog_entry in download_catalog.entries() {
        let huggingface_id = catalog_entry.huggingface_id();
        let exact_copy =
            find_exact_copy(huggingface_id, catalog_entry.revision(), discovered_models);
        let served_copy =
            find_discovered_copy(huggingface_id, catalog_entry.revision(), discovered_models);
        let has_validated_publication = validated_publications.contains(huggingface_id);
        let is_ready = exact_copy.is_some() || has_validated_publication;
        let is_served = is_ready || served_copy.is_some();
        let requestable_model_id = is_served.then(|| {
            served_copy.map_or_else(
                || requestable_model_id_from_huggingface_id(huggingface_id),
                |model| model.model_id.clone(),
            )
        });
        projections.push(CatalogEntryProjection {
            catalog_entry,
            ready_on_this_mac: is_ready,
            served_on_this_mac: is_served,
            discovered_model_directory: served_copy
                .map(|model| model.model_directory.display().to_string()),
            requestable_model_id,
            download_state: current_job
                .filter(|job| job.huggingface_id() == huggingface_id && !is_ready)
                .map(|job| job.state().as_str()),
            capabilities: catalog_entry.capabilities().clone(),
        });
    }
    projections
}

/// Derives the local requestable model ID from the Hugging Face identity's leaf segment.
/// Discovery publishes Library models under their leaf directory name, so "org/Model-Name"
/// becomes requestable as "Model-Name".
pub(crate) fn requestable_model_id_from_huggingface_id(huggingface_id: &str) -> String {
    config::leaf_model_id(huggingface_id).to_owned()
}
