//! Refreshes the one live discovery snapshot after atomic Library publication.

use std::{
    error::Error,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::library::DownloadPublicationRefresh;
use crate::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver, WorkerHandle};

pub struct LibraryModelDiscoveryRefresh {
    runtime_config_resolver: ResolvedRuntimeConfigResolver,
    reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    worker_handle: WorkerHandle,
    runtime_handle: tokio::runtime::Handle,
}

impl LibraryModelDiscoveryRefresh {
    pub fn new(
        runtime_config_resolver: ResolvedRuntimeConfigResolver,
        reloadable_config: Arc<RwLock<ResolvedRuntimeConfig>>,
        worker_handle: WorkerHandle,
    ) -> Self {
        Self {
            runtime_config_resolver,
            reloadable_config,
            worker_handle,
            runtime_handle: tokio::runtime::Handle::current(),
        }
    }
}

impl DownloadPublicationRefresh for LibraryModelDiscoveryRefresh {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        let candidate_config = self.runtime_config_resolver.load()?;
        if !candidate_config
            .discovered_models
            .iter()
            .any(|model| model.model_directory == published_directory)
        {
            return Err(std::io::Error::other(
                "published model did not satisfy executable discovery validation",
            )
            .into());
        }
        self.runtime_handle.block_on(
            self.worker_handle
                .update_model_policy_catalog(Arc::clone(&candidate_config.model_policy_catalog)),
        )?;
        let mut live_config = self
            .reloadable_config
            .write()
            .map_err(|_| std::io::Error::other("live discovery lock was poisoned"))?;
        live_config.discovered_models = candidate_config.discovered_models;
        live_config.model_policy_catalog = candidate_config.model_policy_catalog;
        live_config.unmatched_model_config_ids = candidate_config.unmatched_model_config_ids;
        Ok(())
    }
}
