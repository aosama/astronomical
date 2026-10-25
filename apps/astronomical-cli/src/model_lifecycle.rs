//! Shared model lifecycle for the one-shot CLI verbs. Resolves which model
//! the user wants (`--model` flag, then the daemon's effective default,
//! then the built-in default), checks the requested capability before
//! touching the network, and when the model is not on this Mac yet starts
//! the daemon download and waits for readiness with live stderr progress.
//! One fresh connection per protocol exchange; the whole download wait is
//! bound by a stage timeout so a wedged download never hangs the CLI.

use std::time::Duration;

use astronomical_config::BUILTIN_DEFAULT_MODEL_ID;
use astronomical_ipc_protocol::{
    DaemonCatalogEntry, DaemonDownloadJob, DaemonListedModel, DaemonWorkerStatus,
};

use crate::daemon_probe::{DaemonProbe, DaemonProbeError, DaemonStatusSnapshot};
use crate::formatting;

/// Bound for the whole download-wait stage: local disk writes are slow but
/// finite; generous enough for multi-GB models, short enough that a wedged
/// download cannot hang the calling script forever.
pub const DOWNLOAD_WAIT_STAGE_BOUND: Duration = Duration::from_secs(120);
/// Wait between download status polls: responsive progress without
/// hammering the daemon with status requests.
pub const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Failures of model resolution and preparation. `ModelUnavailable` exits 2
/// (usage: the user asked for something this machine cannot serve); the rest
/// exit 1 as transient daemon problems.
#[derive(Debug, thiserror::Error)]
pub enum ModelLifecycleError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The Astronomical worker is not ready yet — retry in a moment.")]
    WorkerNotReady,
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("{reason}")]
    ModelUnavailable { reason: String },
    #[error("the model download failed: {reason}")]
    DownloadFailed { reason: String },
}

impl From<DaemonProbeError> for ModelLifecycleError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                ModelLifecycleError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => {
                ModelLifecycleError::DaemonStoppedResponding
            }
            DaemonProbeError::DaemonRejected { reason } => {
                ModelLifecycleError::DownloadFailed { reason }
            }
        }
    }
}

/// The capability the verb needs from its model. Checked before download so
/// the CLI never fetches a model that cannot serve the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredCapability {
    Chat,
    Embeddings,
}

impl RequiredCapability {
    /// The capability word and the article that reads correctly before it.
    fn capability_text(self) -> (&'static str, &'static str) {
        match self {
            RequiredCapability::Chat => ("a", "chat"),
            RequiredCapability::Embeddings => ("an", "embeddings"),
        }
    }

    fn describes_installed_model(self, listed_model: &DaemonListedModel) -> bool {
        match self {
            RequiredCapability::Chat => listed_model.context_window.is_some(),
            RequiredCapability::Embeddings => listed_model.supports_embeddings,
        }
    }

    fn describes_catalog_entry(self, catalog_entry: &DaemonCatalogEntry) -> bool {
        match self {
            RequiredCapability::Chat => catalog_entry.context_window.is_some(),
            RequiredCapability::Embeddings => catalog_entry.supports_embeddings,
        }
    }
}

/// Everything the shared lifecycle needs, injected so tests can stub it.
pub struct ModelLifecycle {
    /// The daemon to probe; one connection per exchange.
    pub daemon_probe: DaemonProbe,
    /// Bound for the whole download-wait stage.
    pub download_stage_bound: Duration,
    /// Wait between download status polls.
    pub download_poll_interval: Duration,
}

impl ModelLifecycle {
    /// Worker state, resident model, and effective default model.
    pub async fn status_snapshot(&self) -> Result<DaemonStatusSnapshot, ModelLifecycleError> {
        Ok(self.daemon_probe.status_snapshot().await?)
    }

    /// Resolves the requested model, verifies the capability, and waits for
    /// a not-yet-present model to finish downloading. Returns the model id
    /// to send on the generation request.
    pub async fn prepare_model_id(
        &self,
        requested_model_id: Option<&str>,
        required_capability: RequiredCapability,
        progress: &mut dyn FnMut(&str),
    ) -> Result<String, ModelLifecycleError> {
        let status_snapshot = self.status_snapshot().await?;
        if status_snapshot.worker_status == DaemonWorkerStatus::Unavailable {
            return Err(ModelLifecycleError::WorkerNotReady);
        }
        let installed_models = self.daemon_probe.models_list().await?;
        let resolved_model_id =
            self.resolve_requested_model_id(requested_model_id, &status_snapshot)?;
        let installed_and_capable = installed_models.iter().any(|listed_model| {
            listed_model.model_id == resolved_model_id
                && required_capability.describes_installed_model(listed_model)
        });
        if installed_and_capable {
            return Ok(resolved_model_id);
        }
        // Installed but incapable: say so plainly instead of pretending the
        // model is missing.
        let (article, capability) = required_capability.capability_text();
        if installed_models
            .iter()
            .any(|listed_model| listed_model.model_id == resolved_model_id)
        {
            let mut reason = format!(
                "model {resolved_model_id} is installed on this Mac but is not {article} \
                 {capability} model — it cannot serve this request; run \
                 `astronomical models supported` to see what it does"
            );
            if requested_model_id.is_none() {
                reason
                    .push_str("; pass --model <id> to pick one explicitly instead of the default");
            }
            return Err(ModelLifecycleError::ModelUnavailable { reason });
        }
        let catalog_entry = self
            .daemon_probe
            .catalog()
            .await?
            .into_iter()
            .find(|catalog_entry| catalog_entry_matches(catalog_entry, &resolved_model_id))
            .ok_or_else(|| ModelLifecycleError::ModelUnavailable {
                reason: unknown_model_reason(
                    &resolved_model_id,
                    &installed_models,
                    required_capability,
                ),
            })?;
        if !required_capability.describes_catalog_entry(&catalog_entry) {
            return Err(ModelLifecycleError::ModelUnavailable {
                reason: format!(
                    "model {resolved_model_id} is not {article} {capability} model — it \
                     cannot serve this request; run `astronomical models supported` to \
                     see what it does"
                ),
            });
        }
        if !catalog_entry.ready_on_this_mac {
            self.wait_for_download(&resolved_model_id, progress).await?;
        }
        Ok(resolved_model_id)
    }

    /// Downloads a catalog entry regardless of capability — the
    /// `models download` verb — and waits until it is ready.
    pub async fn ensure_downloaded(
        &self,
        requested_model_id: &str,
        progress: &mut dyn FnMut(&str),
    ) -> Result<String, ModelLifecycleError> {
        let catalog_entry = self
            .daemon_probe
            .catalog()
            .await?
            .into_iter()
            .find(|catalog_entry| catalog_entry_matches(catalog_entry, requested_model_id))
            .ok_or_else(|| ModelLifecycleError::ModelUnavailable {
                reason: format!(
                    "model {requested_model_id} is not in the release catalog; run \
                     `astronomical models supported` to list downloadable models"
                ),
            })?;
        if !catalog_entry.ready_on_this_mac {
            self.wait_for_download(requested_model_id, progress).await?;
        }
        Ok(requested_model_id.to_owned())
    }

    /// Flag > daemon effective default > built-in default. An id the machine
    /// does not know is resolved here and rejected later against the catalog
    /// so the error can list near matches instead of guessing at the flag.
    fn resolve_requested_model_id(
        &self,
        requested_model_id: Option<&str>,
        status_snapshot: &DaemonStatusSnapshot,
    ) -> Result<String, ModelLifecycleError> {
        if let Some(requested_model_id) = requested_model_id {
            return Ok(requested_model_id.to_owned());
        }
        if let Some(default_model_id) = &status_snapshot.default_model_id {
            return Ok(default_model_id.clone());
        }
        Ok(BUILTIN_DEFAULT_MODEL_ID.to_owned())
    }

    /// Starts the download (or resumes a matching paused job) and polls the
    /// job state until the catalog says the model is ready, the job reports
    /// a failure, or the stage bound expires.
    async fn wait_for_download(
        &self,
        model_id: &str,
        progress: &mut dyn FnMut(&str),
    ) -> Result<(), ModelLifecycleError> {
        self.daemon_probe.download_start(model_id).await?;
        let download_deadline = tokio::time::Instant::now() + self.download_stage_bound;
        progress(&format!("downloading {model_id} …"));
        loop {
            tokio::time::sleep(self.download_poll_interval).await;
            let active_job = self.daemon_probe.download_status().await?;
            match active_job {
                Some(active_job) => {
                    if let Some(download_error) = &active_job.error {
                        return Err(ModelLifecycleError::DownloadFailed {
                            reason: format!("{}: {download_error}", active_job.huggingface_id),
                        });
                    }
                    progress(&render_download_progress(&active_job));
                    if tokio::time::Instant::now() >= download_deadline {
                        return Err(ModelLifecycleError::DownloadFailed {
                            reason: download_timeout_reason(model_id),
                        });
                    }
                }
                None => {
                    // The daemon deletes the job file when a download
                    // succeeds, so a vanished job is only believed once the
                    // catalog agrees the model is ready; a vanished job with
                    // no catalog download state means it never really ran.
                    let catalog_entry = self
                        .daemon_probe
                        .catalog()
                        .await?
                        .into_iter()
                        .find(|catalog_entry| catalog_entry_matches(catalog_entry, model_id));
                    match catalog_entry {
                        Some(catalog_entry) if catalog_entry.ready_on_this_mac => {
                            progress(&format!("{model_id} is ready"));
                            return Ok(());
                        }
                        Some(catalog_entry) if catalog_entry.download_state.is_some() => {
                            // Job file was not visible this poll but the
                            // catalog still tracks state: keep waiting.
                        }
                        _ => {
                            return Err(ModelLifecycleError::DownloadFailed {
                                reason: format!(
                                    "the download of {model_id} did not complete; run \
                                     `astronomical models download {model_id}` to retry"
                                ),
                            });
                        }
                    }
                    if tokio::time::Instant::now() >= download_deadline {
                        return Err(ModelLifecycleError::DownloadFailed {
                            reason: download_timeout_reason(model_id),
                        });
                    }
                }
            }
        }
    }
}

fn download_timeout_reason(model_id: &str) -> String {
    format!(
        "the download of {model_id} did not finish in time; run `astronomical status` to \
         check on it"
    )
}

/// Mirrors the daemon's catalog entry matching: the full huggingface id, or
/// the leaf derived from it against the request's leaf. The leaf is derived
/// here rather than taken from the wire's `requestable_model_id`, which
/// stays `None` until the entry is ready on this Mac.
pub(crate) fn catalog_entry_matches(
    catalog_entry: &DaemonCatalogEntry,
    requested_model_id: &str,
) -> bool {
    catalog_entry.huggingface_id == requested_model_id
        || astronomical_config::leaf_model_id(&catalog_entry.huggingface_id)
            == astronomical_config::leaf_model_id(requested_model_id)
}

/// Usage-grade rejection for a model neither installed nor downloadable:
/// near matches from what the machine actually has, plus the pointer to the
/// full downloadable list.
pub(crate) fn unknown_model_reason(
    requested_model_id: &str,
    installed_models: &[DaemonListedModel],
    required_capability: RequiredCapability,
) -> String {
    let installed_ids: Vec<&str> = installed_models
        .iter()
        .map(|listed_model| listed_model.model_id.as_str())
        .collect();
    let near_matches = astronomical_config::near_model_matches(requested_model_id, &installed_ids);
    let (_, capability) = required_capability.capability_text();
    let mut reason = format!(
        "model {requested_model_id} is not installed on this Mac and is not in the release \
         catalog for {capability} models"
    );
    if !near_matches.is_empty() {
        reason.push_str(&format!(" (did you mean: {}?)", near_matches.join(", ")));
    }
    reason.push_str("; run `astronomical models supported` to list what can be downloaded");
    reason
}

/// One live-progress line: the terminal re-uses the line while the download
/// runs. Sizes render in marketed gigabytes (1 GB = 1,073,741,824 bytes).
pub(crate) fn render_download_progress(download_job: &DaemonDownloadJob) -> String {
    if download_job.bytes_total == 0 {
        return format!("{}: {}", download_job.huggingface_id, download_job.state);
    }
    let completed_gigabytes = formatting::format_gigabytes(download_job.bytes_completed);
    let total_gigabytes = formatting::format_gigabytes(download_job.bytes_total);
    let percentage = 100.0 * download_job.bytes_completed as f64 / download_job.bytes_total as f64;
    format!(
        "{}: {} {} GB / {} GB ({:.0}%)",
        download_job.huggingface_id,
        download_job.state,
        completed_gigabytes,
        total_gigabytes,
        percentage,
    )
}
