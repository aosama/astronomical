//! Daemon IPC handlers for the model-lifecycle CLI verbs: installed model
//! listing, catalog projection, download control, and default-model setting.

use std::collections::BTreeSet;
use std::sync::Arc;

use astronomical_config::{
    DiscoveredModel, ModelCapabilities, leaf_model_id, near_model_matches, resolve_model_id,
    write_default_model,
};
use astronomical_ipc_protocol::{
    DaemonCatalogEntry, DaemonDownloadJob, DaemonListedModel, DaemonResponse, DaemonTransportError,
    StreamingResponseWriter,
};

use crate::{
    ipc::service::{
        DaemonIpcGenerationContext, send_terminal_streaming_response,
        unknown_model_rejection_reason,
    },
    library::{
        CatalogEntryProjection, DownloadCatalog, DownloadCatalogEntry, DownloadJob,
        DownloadJobState, LibraryDownloadCoordinator, LibraryDownloadCoordinatorError,
        requestable_model_id_from_huggingface_id,
    },
    performance::attribution::{
        DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
    },
    worker::health::WorkerHealthSnapshot,
};

/// Sends a `RequestRejected` terminal frame with the given reason.
pub(crate) async fn send_request_rejection(
    streaming_response_writer: StreamingResponseWriter,
    reason: String,
) -> Result<(), DaemonTransportError> {
    send_terminal_streaming_response(
        streaming_response_writer,
        &DaemonResponse::RequestRejected { reason },
    )
    .await
}

/// Lists the models discovered on this machine with their resident marker.
pub(super) async fn handle_models_list(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let listed_models = daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcModelsList,
            || async {
                let health_snapshot = generation_context.executor.worker_health_snapshot();
                let discovered_models = live_discovered_models(generation_context);
                build_listed_models(&health_snapshot, &discovered_models)
            },
            |_| DaemonPerformanceMeasurement::success(),
        )
        .await;
    let list_response = DaemonResponse::ModelsList {
        models: listed_models,
    };
    send_terminal_streaming_response(streaming_response_writer, &list_response).await
}

fn build_listed_models(
    health_snapshot: &WorkerHealthSnapshot,
    discovered_models: &[DiscoveredModel],
) -> Vec<DaemonListedModel> {
    discovered_models
        .iter()
        .map(|discovered_model| DaemonListedModel {
            model_id: discovered_model.model_id.clone(),
            family: discovered_model.model_family.as_str().to_owned(),
            context_window: match &discovered_model.capabilities {
                ModelCapabilities::Chat(chat_capabilities) => {
                    Some(chat_capabilities.context_window)
                }
                ModelCapabilities::ImageGeneration(_) | ModelCapabilities::Embeddings(_) => None,
            },
            supports_embeddings: matches!(
                discovered_model.capabilities,
                ModelCapabilities::Embeddings(_)
            ),
            supports_image_generation: matches!(
                &discovered_model.capabilities,
                ModelCapabilities::ImageGeneration(image_capabilities)
                    if image_capabilities.supports_text_to_image
            ),
            is_resident: is_resident_model(health_snapshot, &discovered_model.model_id),
            size_bytes: discovered_model.model_size_bytes,
        })
        .collect()
}

/// The worker drops the previous model before loading the next, while the snapshot
/// still names it until the swap ends, so nothing is resident during a load.
fn is_resident_model(health_snapshot: &WorkerHealthSnapshot, model_id: &str) -> bool {
    health_snapshot.model_load_in_flight.is_none()
        && health_snapshot.ready_model_id.as_deref() == Some(model_id)
}

/// Lists the release download catalog with local readiness per entry.
pub(super) async fn handle_catalog(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let catalog_response = daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcCatalog,
            || async {
                let current_job = active_download_job(generation_context).await;
                let validated_publications =
                    active_validated_publications(generation_context).await;
                let discovered_models = live_discovered_models(generation_context);
                let projections = crate::library::project_catalog_entries(
                    generation_context.download_catalog.as_ref(),
                    &discovered_models,
                    &validated_publications,
                    current_job.as_ref(),
                );
                build_catalog_response(&projections)
            },
            |_| DaemonPerformanceMeasurement::success(),
        )
        .await;
    send_terminal_streaming_response(streaming_response_writer, &catalog_response).await
}

fn build_catalog_response(projections: &[CatalogEntryProjection]) -> DaemonResponse {
    let entries = projections
        .iter()
        .map(|projection| {
            let catalog_entry = projection.catalog_entry;
            let capabilities = &projection.capabilities;
            DaemonCatalogEntry {
                huggingface_id: catalog_entry.huggingface_id().to_owned(),
                display_name: catalog_entry.display_name().to_owned(),
                family: catalog_entry.family().as_str().to_owned(),
                approximate_size_bytes: catalog_entry.approximate_size_bytes(),
                ready_on_this_mac: projection.ready_on_this_mac,
                requestable_model_id: projection.requestable_model_id.clone(),
                download_state: projection.download_state.map(str::to_owned),
                context_window: capabilities.context_window,
                supports_reasoning: capabilities.supports_reasoning,
                supports_vision: capabilities.supports_vision,
                supports_tool_calls: capabilities.supports_tool_calls,
                supports_image_generation: capabilities.supports_image_generation,
                supports_embeddings: capabilities.supports_embeddings,
            }
        })
        .collect();
    DaemonResponse::Catalog { entries }
}

/// Starts a download for the requested model, or resumes a paused job for
/// the same catalog entry. Rejects when a different download is active.
pub(super) async fn handle_download_start(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    model_id: String,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let Some(download_coordinator) = &generation_context.library_download_coordinator else {
        return send_request_rejection(
            streaming_response_writer,
            "the daemon has no Library download coordinator wired".to_owned(),
        )
        .await;
    };
    let Some(catalog_entry) =
        resolve_catalog_entry(generation_context.download_catalog.as_ref(), &model_id)
    else {
        return send_request_rejection(
            streaming_response_writer,
            unknown_model_rejection_reason(
                &model_id,
                &near_model_matches(&model_id, &catalog_model_id_candidates(
                    generation_context.download_catalog.as_ref(),
                )),
                "the model is not in the release catalog; run `astronomical models supported` to list downloadable models",
            ),
        )
        .await;
    };
    let huggingface_id = catalog_entry.huggingface_id().to_owned();
    let start_outcome = daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcDownloadStart,
            || start_or_resume_download(download_coordinator, &huggingface_id),
            |outcome| {
                if outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                }
            },
        )
        .await;
    match start_outcome {
        Ok(()) => {
            send_terminal_streaming_response(
                streaming_response_writer,
                &DaemonResponse::DownloadStarted { huggingface_id },
            )
            .await
        }
        Err(rejection_reason) => {
            send_request_rejection(streaming_response_writer, rejection_reason).await
        }
    }
}

/// Resolves a requestable model id or a full huggingface id to a catalog entry.
fn resolve_catalog_entry<'catalog>(
    download_catalog: &'catalog DownloadCatalog,
    requested_model_id: &str,
) -> Option<&'catalog DownloadCatalogEntry> {
    let requested_leaf = leaf_model_id(requested_model_id);
    download_catalog.entries().iter().find(|catalog_entry| {
        catalog_entry.huggingface_id() == requested_model_id
            || requestable_model_id_from_huggingface_id(catalog_entry.huggingface_id())
                == requested_leaf
    })
}

fn catalog_model_id_candidates(download_catalog: &DownloadCatalog) -> Vec<String> {
    download_catalog
        .entries()
        .iter()
        .map(|catalog_entry| {
            requestable_model_id_from_huggingface_id(catalog_entry.huggingface_id())
        })
        .collect()
}

/// What `DownloadStart` should do, given the job the daemon already holds.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DownloadStartAction {
    /// No job exists: begin a fresh download.
    Start,
    /// The same model's job is paused or failed: continue it from the bytes
    /// already staged on disk.
    Resume,
    /// The same model's job is already running: there is nothing to start,
    /// and the caller should simply keep watching it.
    AlreadyRunning,
    /// A different model's job holds the Library.
    BusyWith { other_huggingface_id: String },
}

/// Decides the start action from the existing job's identity and state.
///
/// `AlreadyRunning` is what makes the CLI safe to rerun. A command that gave
/// up waiting (or was interrupted with Ctrl-C) leaves its download running in
/// the daemon, and the next run asks to start the same download again. That
/// must attach to the running job, not fail with "another download is
/// active", or the user could never finish a large download that took longer
/// than one command was willing to wait.
pub(crate) fn download_start_action(
    existing_job: Option<(&str, DownloadJobState)>,
    requested_huggingface_id: &str,
) -> DownloadStartAction {
    let Some((existing_huggingface_id, existing_state)) = existing_job else {
        return DownloadStartAction::Start;
    };
    if existing_huggingface_id != requested_huggingface_id {
        return DownloadStartAction::BusyWith {
            other_huggingface_id: existing_huggingface_id.to_owned(),
        };
    }
    match existing_state {
        DownloadJobState::Paused | DownloadJobState::Failed => DownloadStartAction::Resume,
        DownloadJobState::CheckingDisk
        | DownloadJobState::FetchingManifest
        | DownloadJobState::Downloading
        | DownloadJobState::Verifying
        | DownloadJobState::Publishing => DownloadStartAction::AlreadyRunning,
    }
}

/// The two facts the start decision needs from a job.
fn download_job_identity(download_job: &DownloadJob) -> (&str, DownloadJobState) {
    (download_job.huggingface_id(), download_job.state())
}

/// How a start or resume attempt ended short of success.
enum StartFailure {
    /// The coordinator refused because another request got there first.
    LostRace(String),
    /// Refused for any other reason.
    Refused(String),
}

/// How long to wait, in total, for a racing request's job to appear: the job
/// file is written a moment after the download task is spawned, so a request
/// that lost the race can briefly find no job at all.
const RACE_SETTLE_ATTEMPTS: u32 = 20;
const RACE_SETTLE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);

async fn start_or_resume_download(
    download_coordinator: &Arc<LibraryDownloadCoordinator>,
    huggingface_id: &str,
) -> Result<(), String> {
    let action = decide_download_start(download_coordinator, huggingface_id).await?;
    match execute_download_start(download_coordinator, huggingface_id, action).await {
        Ok(()) => Ok(()),
        Err(StartFailure::Refused(reason)) => Err(reason),
        Err(StartFailure::LostRace(reason)) => {
            attach_to_the_racing_download(download_coordinator, huggingface_id, reason).await
        }
    }
}

/// Reads the daemon's current job and decides what asking to download this
/// model should do.
async fn decide_download_start(
    download_coordinator: &Arc<LibraryDownloadCoordinator>,
    huggingface_id: &str,
) -> Result<DownloadStartAction, String> {
    let existing_job = match download_coordinator.current_job().await {
        Ok(existing_job) => existing_job,
        Err(query_error) => {
            return Err(format!(
                "the download job state could not be read: {query_error}"
            ));
        }
    };
    let existing_identity = existing_job.as_ref().map(download_job_identity);
    Ok(download_start_action(existing_identity, huggingface_id))
}

async fn execute_download_start(
    download_coordinator: &Arc<LibraryDownloadCoordinator>,
    huggingface_id: &str,
    action: DownloadStartAction,
) -> Result<(), StartFailure> {
    match action {
        DownloadStartAction::AlreadyRunning => Ok(()),
        DownloadStartAction::Resume => match download_coordinator.resume().await {
            Ok(()) => Ok(()),
            Err(resume_error) => Err(start_failure(
                &resume_error,
                format!("the paused download could not be resumed: {resume_error}"),
            )),
        },
        DownloadStartAction::Start => match download_coordinator.start(huggingface_id).await {
            Ok(_) => Ok(()),
            Err(start_error) => Err(start_failure(
                &start_error,
                format!("the download could not be started: {start_error}"),
            )),
        },
        DownloadStartAction::BusyWith {
            other_huggingface_id,
        } => Err(StartFailure::Refused(format!(
            "a download of {other_huggingface_id} is already active; wait for it to finish or cancel it before downloading {huggingface_id}"
        ))),
    }
}

fn start_failure(error: &LibraryDownloadCoordinatorError, reason: String) -> StartFailure {
    match error {
        LibraryDownloadCoordinatorError::LibraryBusy => StartFailure::LostRace(reason),
        _ => StartFailure::Refused(reason),
    }
}

/// Two commands asking for the same model at once (two terminals on a fresh
/// install) both find no job, and the second start is refused as busy. That
/// refusal means someone else is already doing the work, so wait briefly for
/// their job to appear: if it is this model's, attach to it; if it is another
/// model's, or never appears, report the refusal.
async fn attach_to_the_racing_download(
    download_coordinator: &Arc<LibraryDownloadCoordinator>,
    huggingface_id: &str,
    refusal_reason: String,
) -> Result<(), String> {
    for _settle_attempt in 0..RACE_SETTLE_ATTEMPTS {
        match decide_download_start(download_coordinator, huggingface_id).await? {
            DownloadStartAction::AlreadyRunning => return Ok(()),
            DownloadStartAction::BusyWith { .. } => return Err(refusal_reason),
            DownloadStartAction::Start | DownloadStartAction::Resume => {
                tokio::time::sleep(RACE_SETTLE_INTERVAL).await;
            }
        }
    }
    Err(refusal_reason)
}

/// Reports the active library download job, if any.
pub(super) async fn handle_download_status(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let download_job = daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcDownloadStatus,
            || active_download_job(generation_context),
            |_| DaemonPerformanceMeasurement::success(),
        )
        .await;
    let download_response = DaemonResponse::DownloadStatus {
        job: download_job.as_ref().map(download_job_to_wire),
    };
    send_terminal_streaming_response(streaming_response_writer, &download_response).await
}

fn download_job_to_wire(download_job: &DownloadJob) -> DaemonDownloadJob {
    DaemonDownloadJob {
        huggingface_id: download_job.huggingface_id().to_owned(),
        state: download_job.state().as_str().to_owned(),
        bytes_completed: download_job.bytes_completed(),
        bytes_total: download_job.bytes_total(),
        error: download_job
            .error_code()
            .map(|error_code| error_code.as_str().to_owned()),
    }
}

/// Persists the default model after validating it against the catalog or the
/// discovered models, so manual installs can also become the default.
pub(super) async fn handle_default_model_set(
    generation_context: &DaemonIpcGenerationContext,
    daemon_attribution_log: &DaemonPerformanceAttributionLog,
    model_id: String,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let set_outcome = daemon_attribution_log
        .measure_async_operation_best_effort(
            DaemonPerformanceOperation::DaemonIpcDefaultModelSet,
            || async {
                let normalized_model_id = normalize_default_model_id(
                    generation_context.download_catalog.as_ref(),
                    &live_discovered_models(generation_context),
                    &model_id,
                );
                match normalized_model_id {
                    Some(normalized_model_id) => write_default_model(
                        generation_context.instance_paths.state_directory(),
                        Some(&normalized_model_id),
                    )
                    .map(|_| normalized_model_id)
                    .map_err(|persist_error| persist_error.to_string()),
                    None => Err(unknown_default_model_error(&model_id, generation_context)),
                }
            },
            |outcome| {
                if outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                }
            },
        )
        .await;
    match set_outcome {
        Ok(normalized_model_id) => {
            send_terminal_streaming_response(
                streaming_response_writer,
                &DaemonResponse::DefaultModelSet {
                    default_model_id: normalized_model_id,
                },
            )
            .await
        }
        Err(rejection_reason) => {
            send_request_rejection(streaming_response_writer, rejection_reason).await
        }
    }
}

/// Maps the requested id to the canonical requestable id: the catalog entry's
/// requestable id when the id names a catalog entry, the discovered model's
/// id when it names a discovered model, otherwise `None`.
fn normalize_default_model_id(
    download_catalog: &DownloadCatalog,
    discovered_models: &[DiscoveredModel],
    requested_model_id: &str,
) -> Option<String> {
    if let Some(catalog_entry) = resolve_catalog_entry(download_catalog, requested_model_id) {
        return Some(requestable_model_id_from_huggingface_id(
            catalog_entry.huggingface_id(),
        ));
    }
    let known_model_ids: Vec<&str> = discovered_models
        .iter()
        .map(|model| model.model_id.as_str())
        .collect();
    let resolved_model_id = resolve_model_id(requested_model_id, &known_model_ids);
    known_model_ids
        .contains(&resolved_model_id)
        .then(|| resolved_model_id.to_owned())
}

fn unknown_default_model_error(
    model_id: &str,
    generation_context: &DaemonIpcGenerationContext,
) -> String {
    let candidates = catalog_model_id_candidates(generation_context.download_catalog.as_ref());
    unknown_model_rejection_reason(
        model_id,
        &near_model_matches(model_id, &candidates),
        "the model is neither in the release catalog nor installed on this machine; set a model from `astronomical models supported` or `astronomical models list`",
    )
}

/// Discovered models from the live config, empty when no config is wired.
pub(crate) fn live_discovered_models(
    generation_context: &DaemonIpcGenerationContext,
) -> Vec<DiscoveredModel> {
    generation_context
        .reloadable_config
        .as_ref()
        .and_then(|live_config| live_config.read().ok())
        .map(|live_config| live_config.discovered_models.clone())
        .unwrap_or_default()
}

/// The active download job, or `None` when no coordinator is wired or no job
/// is in flight.
async fn active_download_job(
    generation_context: &DaemonIpcGenerationContext,
) -> Option<DownloadJob> {
    match &generation_context.library_download_coordinator {
        Some(download_coordinator) => download_coordinator.current_job().await.ok().flatten(),
        None => None,
    }
}

async fn active_validated_publications(
    generation_context: &DaemonIpcGenerationContext,
) -> BTreeSet<String> {
    match &generation_context.library_download_coordinator {
        Some(download_coordinator) => download_coordinator.validated_publications_snapshot().await,
        None => Default::default(),
    }
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod models_tests;
