//! Refreshes the one live discovery snapshot after Download Models publishes a
//! model or moves one to the Trash.

use std::{
    error::Error,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::library::DownloadPublicationRefresh;
use crate::{
    ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver, RuntimeModelPolicy, WorkerHandle,
};

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
        // An added model gets its policies before it becomes requestable, so a
        // request never names a model the worker has no policy for.
        self.update_worker_policies(Arc::clone(&candidate_config.model_policy_catalog))?;
        self.make_discovery_live(candidate_config)
    }

    fn refresh_after_removal(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let candidate_config = self.runtime_config_resolver.load()?;
        let remaining_model_policies = Arc::clone(&candidate_config.model_policy_catalog);
        // The reverse order for a removed model. Its files are already in the
        // Trash when this runs, so a request that arrives before this refresh
        // can still pick it and fail its load; unlisting it first ends that
        // window as early as possible, and only then does the worker drop its
        // policy, so no listed model is ever without one.
        self.make_discovery_live(candidate_config)?;
        self.update_worker_policies(remaining_model_policies)
    }
}

impl LibraryModelDiscoveryRefresh {
    fn update_worker_policies(
        &self,
        model_policy_catalog: Arc<std::collections::HashMap<String, RuntimeModelPolicy>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.runtime_handle.block_on(
            self.worker_handle
                .update_model_policy_catalog(model_policy_catalog),
        )?;
        Ok(())
    }

    fn make_discovery_live(
        &self,
        candidate_config: ResolvedRuntimeConfig,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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
