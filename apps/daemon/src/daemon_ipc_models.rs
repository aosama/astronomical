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
    daemon_ipc::{
        DaemonIpcGenerationContext, send_terminal_streaming_response,
        unknown_model_rejection_reason,
    },
    library::{
        CatalogEntryProjection, DownloadCatalog, DownloadCatalogEntry, DownloadJob,
        LibraryDownloadCoordinator, requestable_model_id_from_huggingface_id,
    },
    supervisor_performance_attribution::{
        SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
        SupervisorPerformanceOperation,
    },
    worker_health::WorkerHealthSnapshot,
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
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let listed_models = supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcModelsList,
            || async {
                let health_snapshot = generation_context.executor.worker_health_snapshot();
                let discovered_models = live_discovered_models(generation_context);
                build_listed_models(&health_snapshot, &discovered_models)
            },
            |_| SupervisorPerformanceMeasurement::success(),
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
            is_resident: health_snapshot.ready_model_id.as_deref()
                == Some(discovered_model.model_id.as_str()),
            size_bytes: discovered_model.model_size_bytes,
        })
        .collect()
}

/// Lists the release download catalog with local readiness per entry.
pub(super) async fn handle_catalog(
    generation_context: &DaemonIpcGenerationContext,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let catalog_response = supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcCatalog,
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
            |_| SupervisorPerformanceMeasurement::success(),
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
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
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
    let start_outcome = supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcDownloadStart,
            || start_or_resume_download(download_coordinator, &huggingface_id),
            |outcome| {
                if outcome.is_ok() {
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
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

async fn start_or_resume_download(
    download_coordinator: &Arc<LibraryDownloadCoordinator>,
    huggingface_id: &str,
) -> Result<(), String> {
    match download_coordinator.current_job().await {
        Ok(Some(active_job)) if active_job.huggingface_id() == huggingface_id => {
            download_coordinator
                .resume()
                .await
                .map(|_| ())
                .map_err(|resume_error| {
                    format!("the paused download could not be resumed: {resume_error}")
                })
        }
        Ok(Some(active_job)) => Err(format!(
            "a download of {} is already active; wait for it to finish or cancel it before downloading {huggingface_id}",
            active_job.huggingface_id()
        )),
        Ok(None) => download_coordinator
            .start(huggingface_id)
            .await
            .map(|_| ())
            .map_err(|start_error| format!("the download could not be started: {start_error}")),
        Err(query_error) => Err(format!(
            "the download job state could not be read: {query_error}"
        )),
    }
}

/// Reports the active library download job, if any.
pub(super) async fn handle_download_status(
    generation_context: &DaemonIpcGenerationContext,
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let download_job = supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcDownloadStatus,
            || active_download_job(generation_context),
            |_| SupervisorPerformanceMeasurement::success(),
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
    supervisor_attribution_log: &SupervisorPerformanceAttributionLog,
    model_id: String,
    streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let set_outcome = supervisor_attribution_log
        .measure_async_operation_best_effort(
            SupervisorPerformanceOperation::DaemonIpcDefaultModelSet,
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
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
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
fn live_discovered_models(generation_context: &DaemonIpcGenerationContext) -> Vec<DiscoveredModel> {
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
