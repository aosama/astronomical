//! Shared projection of the release catalog joined with local readiness state.
//!
//! Both the REST catalog endpoint and the daemon IPC `Catalog` request answer
//! the same question — "is this catalog entry installed on this Mac, and under
//! which requestable ID?" — so the join logic lives here once.

use std::collections::BTreeSet;

use astronomical_config::DiscoveredModel;

use crate::library::{
    DownloadCatalog, DownloadCatalogCapabilities, DownloadCatalogEntry, DownloadJob,
};

/// Readiness state of one catalog entry, computed once per request.
pub(crate) struct CatalogEntryProjection<'a> {
    pub catalog_entry: &'a DownloadCatalogEntry,
    pub ready_on_this_mac: bool,
    pub discovered_model_directory: Option<String>,
    pub requestable_model_id: Option<String>,
    pub download_state: Option<&'static str>,
    pub capabilities: DownloadCatalogCapabilities,
}

/// Joins every catalog entry with discovery and publication state.
///
/// An entry is ready when the model is discovered (provider identity and
/// revision both match) or when a validated download publication exists for
/// its Hugging Face identity.
pub(crate) fn project_catalog_entries<'a>(
    download_catalog: &'a DownloadCatalog,
    discovered_models: &[DiscoveredModel],
    validated_publications: &BTreeSet<String>,
    current_job: Option<&DownloadJob>,
) -> Vec<CatalogEntryProjection<'a>> {
    let mut projections = Vec::with_capacity(download_catalog.entry_count());
    for catalog_entry in download_catalog.entries() {
        let huggingface_id = catalog_entry.huggingface_id();
        let discovered_model = discovered_models.iter().find(|model| {
            model.provider_model_id.as_deref() == Some(huggingface_id)
                && model.revision == catalog_entry.revision()
        });
        let has_validated_publication = validated_publications.contains(huggingface_id);
        let is_ready = discovered_model.is_some() || has_validated_publication;
        let requestable_model_id = is_ready.then(|| {
            discovered_model.map_or_else(
                || requestable_model_id_from_huggingface_id(huggingface_id),
                |model| model.model_id.clone(),
            )
        });
        projections.push(CatalogEntryProjection {
            catalog_entry,
            ready_on_this_mac: is_ready,
            discovered_model_directory: discovered_model
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
    huggingface_id
        .rsplit('/')
        .next()
        .unwrap_or(huggingface_id)
        .to_owned()
}
