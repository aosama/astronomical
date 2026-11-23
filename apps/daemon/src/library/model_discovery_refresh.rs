//! Refreshes the one live discovery snapshot after Download Models publishes a
//! model or moves one to the Trash, and after a person adds, moves or removes
//! a model folder by hand (`model_folder_watch` notices those).

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
    /// Held from discovery to publication by every refresh. A download's
    /// refresh and the folder watch's rescan can otherwise interleave, and a
    /// rescan that discovered before a download published would then list
    /// its older view over the newer one, unlisting the model just published.
    refresh_in_progress: std::sync::Mutex<()>,
}

type ModelPolicyCatalog = Arc<std::collections::HashMap<String, RuntimeModelPolicy>>;

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
            refresh_in_progress: std::sync::Mutex::new(()),
        }
    }
}

impl DownloadPublicationRefresh for LibraryModelDiscoveryRefresh {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _refresh_turn = self.take_refresh_turn()?;
        self.publish_downloaded_model(published_directory)
    }

    fn refresh_after_removal(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _refresh_turn = self.take_refresh_turn()?;
        self.unpublish_removed_model()
    }
}

impl LibraryModelDiscoveryRefresh {
    fn publish_downloaded_model(
        &self,
        published_directory: &Path,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
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

    fn unpublish_removed_model(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
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

/// What a rescan changed in the listed models, by model ID.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ModelLibraryChange {
    pub added_model_ids: Vec<String>,
    pub removed_model_ids: Vec<String>,
    /// Models still listed whose folder or settings moved or changed.
    pub changed_model_ids: Vec<String>,
}

impl ModelLibraryChange {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added_model_ids.is_empty()
            && self.removed_model_ids.is_empty()
            && self.changed_model_ids.is_empty()
    }
}

/// The order a rescan publishes in, so no listed model is ever without a
/// policy and no model gets listed before its policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RescanPublicationOrder {
    /// Models were only gained: their policies first, then the list.
    PoliciesThenList,
    /// Models were only lost or changed: the list first, then the policies.
    ListThenPolicies,
    /// Models were gained and lost: policies for both, then the list, then
    /// the new policies alone.
    BridgingPoliciesThenListThenPolicies,
}

pub(crate) fn rescan_publication_order(
    library_change: &ModelLibraryChange,
) -> RescanPublicationOrder {
    match (
        library_change.added_model_ids.is_empty(),
        library_change.removed_model_ids.is_empty(),
    ) {
        (false, true) => RescanPublicationOrder::PoliciesThenList,
        (false, false) => RescanPublicationOrder::BridgingPoliciesThenListThenPolicies,
        (true, _) => RescanPublicationOrder::ListThenPolicies,
    }
}

impl LibraryModelDiscoveryRefresh {
    /// Discovers the library again and makes the result live, in the order
    /// `rescan_publication_order` gives.
    pub fn rescan(&self) -> Result<ModelLibraryChange, Box<dyn Error + Send + Sync>> {
        let _refresh_turn = self.take_refresh_turn()?;
        let candidate_config = self.runtime_config_resolver.load()?;
        let (live_models, live_policies) = {
            let live_config = self
                .reloadable_config
                .read()
                .map_err(|_| std::io::Error::other("live discovery lock was poisoned"))?;
            (
                live_config.discovered_models.clone(),
                Arc::clone(&live_config.model_policy_catalog),
            )
        };
        let library_change =
            model_library_change(&live_models, &candidate_config.discovered_models);
        if library_change.is_empty() {
            return Ok(library_change);
        }
        let candidate_policies = Arc::clone(&candidate_config.model_policy_catalog);
        match rescan_publication_order(&library_change) {
            RescanPublicationOrder::PoliciesThenList => {
                self.update_worker_policies(candidate_policies)?;
                self.make_discovery_live(candidate_config)?;
            }
            RescanPublicationOrder::ListThenPolicies => {
                self.make_discovery_live(candidate_config)?;
                self.update_worker_policies(candidate_policies)?;
            }
            RescanPublicationOrder::BridgingPoliciesThenListThenPolicies => {
                let bridging_policies = policies_for_both(&live_policies, &candidate_policies);
                self.update_worker_policies(bridging_policies)?;
                self.make_discovery_live(candidate_config)?;
                self.update_worker_policies(candidate_policies)?;
            }
        }
        Ok(library_change)
    }

    fn take_refresh_turn(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, ()>, Box<dyn Error + Send + Sync>> {
        self.refresh_in_progress
            .lock()
            .map_err(|_| std::io::Error::other("the discovery refresh lock was poisoned").into())
    }

    fn update_worker_policies(
        &self,
        model_policy_catalog: ModelPolicyCatalog,
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

/// Every policy of both catalogs, the newer winning for a model in both, so
/// the worker holds a policy for every model listed before and after.
fn policies_for_both(
    live_policies: &ModelPolicyCatalog,
    candidate_policies: &ModelPolicyCatalog,
) -> ModelPolicyCatalog {
    let mut bridging_policies = live_policies.as_ref().clone();
    for (model_id, candidate_policy) in candidate_policies.iter() {
        bridging_policies.insert(model_id.clone(), candidate_policy.clone());
    }
    Arc::new(bridging_policies)
}

/// The difference between two discovery snapshots, by model ID.
pub(crate) fn model_library_change(
    live_models: &[config::DiscoveredModel],
    candidate_models: &[config::DiscoveredModel],
) -> ModelLibraryChange {
    let find_model = |models: &[config::DiscoveredModel], model_id: &str| {
        models
            .iter()
            .find(|model| model.model_id == model_id)
            .cloned()
    };
    let mut library_change = ModelLibraryChange::default();
    for candidate_model in candidate_models {
        match find_model(live_models, &candidate_model.model_id) {
            None => library_change
                .added_model_ids
                .push(candidate_model.model_id.clone()),
            Some(live_model) if live_model != *candidate_model => library_change
                .changed_model_ids
                .push(candidate_model.model_id.clone()),
            Some(_) => {}
        }
    }
    for live_model in live_models {
        if find_model(candidate_models, &live_model.model_id).is_none() {
            library_change
                .removed_model_ids
                .push(live_model.model_id.clone());
        }
    }
    library_change
}

#[cfg(test)]
#[path = "model_discovery_refresh_tests.rs"]
mod tests;
