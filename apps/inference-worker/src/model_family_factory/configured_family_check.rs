//! The worker's check that the policy it was sent names the family it finds
//! on disk.

use std::path::PathBuf;

use config::{ModelFamily, classify_model_directory};
use ipc_protocol::WorkerModelConfiguration;

/// Why a folder is refused when the policy sent for it names another family.
pub(super) const FAMILY_MISMATCH_REASON: &str =
    "selected model configuration does not match its classified model family";

/// The daemon names the family discovery found; a folder that classifies as
/// another family by the time the worker loads it is refused before any
/// engine is built. A chat policy names no family, so only its kind is
/// checked, by the factory's match.
pub(super) fn ensure_configuration_names_classified_family(
    model_family: Option<ModelFamily>,
    model_configuration: &WorkerModelConfiguration,
) -> Result<(), String> {
    match (model_configuration.model_family(), model_family) {
        (Some(configured_family), Some(classified_family))
            if configured_family != classified_family.as_str() =>
        {
            Err(FAMILY_MISMATCH_REASON.to_owned())
        }
        _ => Ok(()),
    }
}

/// Classifies the folder off the async runtime and refuses it when the
/// policy names another family. `None` is a folder of no known family,
/// which the factory refuses itself.
pub(super) async fn classified_model_family(
    model_directory_path: PathBuf,
    model_configuration: &WorkerModelConfiguration,
) -> Result<Option<ModelFamily>, String> {
    let model_family = tokio::task::spawn_blocking(move || {
        classify_model_directory(&model_directory_path)
            .map_err(|_| "selected model family could not be classified".to_owned())
    })
    .await
    .map_err(|_| "model-family classification task failed".to_owned())??;
    ensure_configuration_names_classified_family(model_family, model_configuration)?;
    Ok(model_family)
}

#[cfg(test)]
#[path = "configured_family_check_tests.rs"]
mod configured_family_check_tests;
