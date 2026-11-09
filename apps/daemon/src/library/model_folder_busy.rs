//! Whether the worker is using a model's folder right now, so Remove from disk
//! never moves files out from under a loaded or loading model.
//!
//! The download coordinator asks this under its command lock, immediately
//! before the move, rather than the HTTP handler asking first: a check made
//! before the lock could pass, then a load could start while the request
//! waited for the lock. A load that starts after the move finds the files gone
//! and fails like any load of a missing model; the discovery refresh that
//! follows the move stops offering the model at all.

use std::path::Path;

use astronomical_config::DiscoveredModel;
use astronomical_ipc_protocol::ModelLoadInFlight;

use crate::application::ApplicationState;
use crate::library::requestable_model_id_from_huggingface_id;

/// Why a model's folder cannot move right now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFolderBusy {
    Loaded,
    Loading,
}

/// Something that can tell whether the worker is using a model folder.
pub trait ModelFolderBusyCheck: Send + Sync {
    fn busy_reason(&self, model_folder: &Path, huggingface_id: &str) -> Option<ModelFolderBusy>;
}

/// The live check: the worker's health and the current discovery.
pub(crate) struct WorkerModelFolderBusyCheck {
    application_state: ApplicationState,
}

impl WorkerModelFolderBusyCheck {
    pub(crate) fn new(application_state: ApplicationState) -> Self {
        Self { application_state }
    }
}

impl ModelFolderBusyCheck for WorkerModelFolderBusyCheck {
    fn busy_reason(&self, model_folder: &Path, huggingface_id: &str) -> Option<ModelFolderBusy> {
        let worker_health_snapshot = self
            .application_state
            .generation_executor
            .worker_health_snapshot();
        let mut model_ids_in_folder = model_ids_served_from_folder(
            &self.application_state.discovered_models_snapshot(),
            model_folder,
        );
        // A freshly published model is requestable under this id before the
        // next discovery names it, so it counts as served from the folder too.
        model_ids_in_folder.push(requestable_model_id_from_huggingface_id(huggingface_id));
        model_folder_busy_reason(
            &model_ids_in_folder,
            worker_health_snapshot.ready_model_id.as_deref(),
            worker_health_snapshot.model_load_in_flight.as_ref(),
        )
    }
}

/// The model ids discovery serves from files inside `model_folder`. Matching by
/// folder rather than by catalog identity also catches a model whose files are
/// in this folder but whose served name differs from the catalog's.
fn model_ids_served_from_folder(
    discovered_models: &[DiscoveredModel],
    model_folder: &Path,
) -> Vec<String> {
    discovered_models
        .iter()
        .filter(|discovered_model| discovered_model.model_directory.starts_with(model_folder))
        .map(|discovered_model| discovered_model.model_id.clone())
        .collect()
}

/// A load that cannot name its model may be loading this one, so it refuses
/// too; the person retries a moment later instead of moving files mid-load.
fn model_folder_busy_reason(
    model_ids_in_folder: &[String],
    ready_model_id: Option<&str>,
    model_load_in_flight: Option<&ModelLoadInFlight>,
) -> Option<ModelFolderBusy> {
    let is_in_folder = |model_id: &str| {
        model_ids_in_folder
            .iter()
            .any(|folder_model_id| folder_model_id == model_id)
    };
    if let Some(load_in_flight) = model_load_in_flight {
        match load_in_flight.model_id.as_deref() {
            None => return Some(ModelFolderBusy::Loading),
            Some(loading_model_id) if is_in_folder(loading_model_id) => {
                return Some(ModelFolderBusy::Loading);
            }
            Some(_) => {}
        }
    }
    ready_model_id
        .filter(|model_id| is_in_folder(model_id))
        .map(|_| ModelFolderBusy::Loaded)
}

#[cfg(test)]
#[path = "model_folder_busy_tests.rs"]
mod model_folder_busy_tests;
