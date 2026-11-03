//! Shared model lifecycle for the one-shot CLI verbs. Resolves which model
//! the user wants (`--model` flag, then the daemon's effective default,
//! then the built-in default), checks the requested capability before
//! touching the network, and when the model is not on this Mac yet starts
//! the daemon download and waits for readiness with live stderr progress.
//! One fresh connection per protocol exchange; the whole download wait is
//! bound by a stage timeout so a wedged download never hangs the CLI.

use std::time::Duration;

use astronomical_config::{BUILTIN_DEFAULT_IMAGE_MODEL_ID, BUILTIN_DEFAULT_MODEL_ID};
use astronomical_ipc_protocol::{
    DaemonCatalogEntry, DaemonDownloadJob, DaemonListedModel, DaemonWorkerStatus,
};

use crate::formatting;
use crate::transport::ipc::{DaemonProbe, DaemonProbeError, DaemonStatusSnapshot};
use download_progress::{DownloadProgressTracker, StallVerdict, StallWatch};

/// How long a download may show no progress at all (no new bytes, no state
/// change) before the wait gives up. It is a bound on silence, not on total
/// time: a 15 GB model takes many minutes on an ordinary connection, and a
/// download that keeps moving is allowed to finish. A wedged download still
/// cannot hang the calling script forever, because it stops moving.
pub const DOWNLOAD_STALL_BOUND: Duration = Duration::from_secs(90);
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

/// Receives one live download-progress line at a time.
///
/// A trait rather than a bare callback so a verb can supply a plain struct
/// that owns its output stream. Any `FnMut(&str)` still qualifies, so a verb
/// that only wants to count or collect lines can pass a callback unchanged.
pub trait ProgressSink {
    fn report(&mut self, progress_line: &str);
}

impl<Callback: FnMut(&str)> ProgressSink for Callback {
    fn report(&mut self, progress_line: &str) {
        self(progress_line);
    }
}

/// The capability the verb needs from its model. Checked before download so
/// the CLI never fetches a model that cannot serve the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredCapability {
    Chat,
    Embeddings,
    ImageGeneration,
}

impl RequiredCapability {
    /// The capability word and the article that reads correctly before it.
    fn capability_text(self) -> (&'static str, &'static str) {
        match self {
            RequiredCapability::Chat => ("a", "chat"),
            RequiredCapability::Embeddings => ("an", "embeddings"),
            RequiredCapability::ImageGeneration => ("an", "image-generation"),
        }
    }

    fn describes_installed_model(self, listed_model: &DaemonListedModel) -> bool {
        match self {
            RequiredCapability::Chat => listed_model.context_window.is_some(),
            RequiredCapability::Embeddings => listed_model.supports_embeddings,
            RequiredCapability::ImageGeneration => listed_model.supports_image_generation,
        }
    }

    fn describes_catalog_entry(self, catalog_entry: &DaemonCatalogEntry) -> bool {
        match self {
            RequiredCapability::Chat => catalog_entry.context_window.is_some(),
            RequiredCapability::Embeddings => catalog_entry.supports_embeddings,
            RequiredCapability::ImageGeneration => catalog_entry.supports_image_generation,
        }
    }
}

/// Everything the shared lifecycle needs, injected so tests can stub it.
pub struct ModelLifecycle {
    /// The daemon to probe; one connection per exchange.
    pub daemon_probe: DaemonProbe,
    /// How long a download may go without any change before the wait gives up.
    pub download_stall_bound: Duration,
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
        progress: &mut dyn ProgressSink,
    ) -> Result<String, ModelLifecycleError> {
        let status_snapshot = self.status_snapshot().await?;
        if status_snapshot.worker_status == DaemonWorkerStatus::Unavailable {
            return Err(ModelLifecycleError::WorkerNotReady);
        }
        let installed_models = self.daemon_probe.models_list().await?;
        let resolved_model_id = self.resolve_requested_model_id(
            requested_model_id,
            required_capability,
            &status_snapshot,
        )?;
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
            self.wait_for_download(&resolved_model_id, &catalog_entry, progress)
                .await?;
        }
        Ok(resolved_model_id)
    }

    /// Downloads a catalog entry regardless of capability — the
    /// `models download` verb — and waits until it is ready.
    pub async fn ensure_downloaded(
        &self,
        requested_model_id: &str,
        progress: &mut dyn ProgressSink,
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
            self.wait_for_download(requested_model_id, &catalog_entry, progress)
                .await?;
        }
        Ok(requested_model_id.to_owned())
    }

    /// Flag > daemon effective default > built-in default. An id the machine
    /// does not know is resolved here and rejected later against the catalog
    /// so the error can list near matches instead of guessing at the flag.
    ///
    /// Images skip the daemon default: it names a chat model, which cannot
    /// draw, so an unflagged `image` goes straight to the built-in image
    /// model instead of failing against a default the user never chose for it.
    fn resolve_requested_model_id(
        &self,
        requested_model_id: Option<&str>,
        required_capability: RequiredCapability,
        status_snapshot: &DaemonStatusSnapshot,
    ) -> Result<String, ModelLifecycleError> {
        if let Some(requested_model_id) = requested_model_id {
            return Ok(requested_model_id.to_owned());
        }
        if required_capability == RequiredCapability::ImageGeneration {
            return Ok(BUILTIN_DEFAULT_IMAGE_MODEL_ID.to_owned());
        }
        if let Some(default_model_id) = &status_snapshot.default_model_id {
            return Ok(default_model_id.clone());
        }
        Ok(BUILTIN_DEFAULT_MODEL_ID.to_owned())
    }

    /// Starts the download (or attaches to one already running, or resumes a
    /// paused or failed one from the bytes on disk) and follows it until the
    /// catalog says the model is ready, the job reports a failure, or it stops
    /// making progress. The elapsed time, transfer rate and time remaining go
    /// to `progress` on every poll.
    async fn wait_for_download(
        &self,
        model_id: &str,
        catalog_entry: &DaemonCatalogEntry,
        progress: &mut dyn ProgressSink,
    ) -> Result<(), ModelLifecycleError> {
        self.daemon_probe.download_start(model_id).await?;
        progress.report(&format!(
            "downloading {model_id} ({} GB) …",
            formatting::format_gigabytes(catalog_entry.approximate_size_bytes)
        ));
        let wait_started_at = tokio::time::Instant::now();
        let mut download_wait = DownloadWait {
            tracker: DownloadProgressTracker::new(),
            stall_watch: StallWatch::new(self.download_stall_bound),
            daemon_was_unreachable: false,
        };
        loop {
            tokio::time::sleep(self.download_poll_interval).await;
            let elapsed = wait_started_at.elapsed();
            let wait_outcome = self
                .poll_download(model_id, elapsed, &mut download_wait, progress)
                .await?;
            if wait_outcome == WaitOutcome::Ready {
                return Ok(());
            }
        }
    }

    /// One poll. A daemon that cannot be reached is not a failed download: an
    /// app update or restart in the middle of a 15 GB transfer is ordinary, and
    /// the daemon resumes from the bytes on disk, so the wait keeps going under
    /// the same stall bound. When the daemon comes back it has parked the job
    /// as paused, so the request to download is repeated to continue it.
    async fn poll_download(
        &self,
        model_id: &str,
        elapsed: Duration,
        download_wait: &mut DownloadWait,
        progress: &mut dyn ProgressSink,
    ) -> Result<WaitOutcome, ModelLifecycleError> {
        let active_job = match self.daemon_probe.download_status().await {
            Ok(active_job) => active_job,
            Err(probe_error @ DaemonProbeError::DaemonRejected { .. }) => {
                return Err(probe_error.into());
            }
            Err(_daemon_unreachable) => {
                return download_wait.observe_unreachable_daemon(elapsed, model_id, progress);
            }
        };
        if download_wait.daemon_was_unreachable {
            download_wait.daemon_was_unreachable = false;
            self.daemon_probe.download_start(model_id).await?;
        }
        match active_job {
            Some(active_job) => download_wait.observe_active_job(&active_job, elapsed, progress),
            None => {
                self.observe_vanished_job(model_id, elapsed, download_wait, progress)
                    .await
            }
        }
    }

    /// The daemon deletes the job file when a download succeeds, so a vanished
    /// job is only believed once the catalog agrees the model is ready. A
    /// vanished job the catalog still tracks is a hand-off in progress and is
    /// waited out under the same stall bound; one it does not track never
    /// really ran.
    async fn observe_vanished_job(
        &self,
        model_id: &str,
        elapsed: Duration,
        download_wait: &mut DownloadWait,
        progress: &mut dyn ProgressSink,
    ) -> Result<WaitOutcome, ModelLifecycleError> {
        let mut matching_entry = None;
        for catalog_entry in self.daemon_probe.catalog().await? {
            if catalog_entry_matches(&catalog_entry, model_id) {
                matching_entry = Some(catalog_entry);
                break;
            }
        }
        match matching_entry {
            Some(catalog_entry) if catalog_entry.ready_on_this_mac => {
                progress.report(&format!("{model_id} is ready"));
                Ok(WaitOutcome::Ready)
            }
            Some(catalog_entry) if catalog_entry.download_state.is_some() => {
                download_wait.check_stall(elapsed, "settling", 0, model_id)?;
                Ok(WaitOutcome::KeepWaiting)
            }
            _ => Err(ModelLifecycleError::DownloadFailed {
                reason: format!(
                    "the download of {model_id} did not complete; run the same command again to \
                     resume it, or `astronomical models download {model_id}`"
                ),
            }),
        }
    }
}

/// What one poll concluded.
#[derive(Debug, Eq, PartialEq)]
enum WaitOutcome {
    KeepWaiting,
    Ready,
}

/// The state one download wait carries between polls.
struct DownloadWait {
    tracker: DownloadProgressTracker,
    stall_watch: StallWatch,
    /// The previous poll found no daemon to talk to.
    daemon_was_unreachable: bool,
}

impl DownloadWait {
    /// Reports one poll of a live job, fails on a job error, and ends the wait
    /// when the job stops making progress.
    fn observe_active_job(
        &mut self,
        active_job: &DaemonDownloadJob,
        elapsed: Duration,
        progress: &mut dyn ProgressSink,
    ) -> Result<WaitOutcome, ModelLifecycleError> {
        if let Some(download_error) = &active_job.error {
            return Err(ModelLifecycleError::DownloadFailed {
                reason: format!(
                    "{}: {download_error}; run the same command again to resume from the bytes \
                     already downloaded",
                    active_job.huggingface_id
                ),
            });
        }
        self.tracker.record(elapsed, active_job.bytes_completed);
        progress.report(&self.tracker.render_line(active_job));
        self.check_stall(
            elapsed,
            &active_job.state,
            active_job.bytes_completed,
            &active_job.huggingface_id,
        )?;
        Ok(WaitOutcome::KeepWaiting)
    }

    /// A poll that found no daemon counts as no progress: it ends the wait only
    /// if the daemon stays away for the whole stall bound.
    fn observe_unreachable_daemon(
        &mut self,
        elapsed: Duration,
        model_id: &str,
        progress: &mut dyn ProgressSink,
    ) -> Result<WaitOutcome, ModelLifecycleError> {
        self.daemon_was_unreachable = true;
        progress.report("waiting for the daemon to come back …");
        self.check_stall(elapsed, "the daemon was unreachable", 0, model_id)?;
        Ok(WaitOutcome::KeepWaiting)
    }

    fn check_stall(
        &mut self,
        elapsed: Duration,
        state: &str,
        bytes_completed: u64,
        model_id: &str,
    ) -> Result<(), ModelLifecycleError> {
        match self.stall_watch.observe(elapsed, state, bytes_completed) {
            StallVerdict::Progressing => Ok(()),
            StallVerdict::Stalled { stalled_for, state } => {
                Err(ModelLifecycleError::DownloadFailed {
                    reason: stall_reason(model_id, stalled_for, &state),
                })
            }
        }
    }
}

/// The wait ended because nothing moved, not because the download failed: the
/// bytes already on disk are kept, so running the command again continues it.
fn stall_reason(model_id: &str, stalled_for: Duration, state: &str) -> String {
    format!(
        "the download of {model_id} made no progress for {}s while {}; run the same command \
         again to continue it from the bytes already on disk, or `astronomical status` to \
         look at it",
        stalled_for.as_secs(),
        state.replace('_', " ")
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

#[path = "model_lifecycle/download_progress.rs"]
mod download_progress;

#[cfg(test)]
mod model_lifecycle_tests;
