//! Finding the folder a decision model is served from.
//!
//! Calibration copies the model the daemon serves, so it must find that
//! folder the way the daemon does: the instance's own models directory first,
//! then the folders the instance configuration lists. Sharing
//! `discover_effective_models` with the daemon keeps the two from disagreeing
//! about which folder a model ID means.

use std::io;
use std::path::{Path, PathBuf};

use config::{
    AstronomicalConfig, AstronomicalInstancePaths, ModelFamily, discover_effective_models,
};
use thiserror::Error;

/// Where a model lives, and every folder the daemon scans for models.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFolderLookup {
    pub model_directory: PathBuf,
    /// Which decision family the model is, which decides what training may change.
    pub model_family: ModelFamily,
    pub scanned_model_roots: Vec<PathBuf>,
    /// Every model ID the daemon serves, so a new folder's name can be
    /// checked against them: a folder named like an installed model would
    /// make that ID ambiguous.
    pub installed_model_ids: Vec<String>,
    /// Other models served from inside `model_directory`, relative to it,
    /// such as the multilingual checkpoint a Laya repository bundles in a
    /// subfolder. The calibration is fitted to one model only, so these are
    /// left out of the copy rather than served again under a calibrated name.
    pub nested_model_directories: Vec<PathBuf>,
}

#[derive(Debug, Error)]
pub enum ModelFolderLookupError {
    #[error("the instance configuration at {config_file_path} cannot be read: {cause}")]
    ConfigUnreadable {
        config_file_path: PathBuf,
        cause: String,
    },
    #[error("the installed models could not be listed: {cause}")]
    DiscoveryFailed { cause: String },
    #[error("{model_id} is not installed in a folder this instance serves from")]
    ModelNotFound { model_id: String },
}

/// The folder `model_id` is served from on this instance.
pub fn find_model_folder(
    instance_paths: &AstronomicalInstancePaths,
    model_id: &str,
) -> Result<ModelFolderLookup, ModelFolderLookupError> {
    let configured_model_roots = configured_model_roots(instance_paths)?;
    let automatic_models_directory = instance_paths.models_directory();
    let effective_discovery =
        match discover_effective_models(&automatic_models_directory, &configured_model_roots) {
            Ok(effective_discovery) => effective_discovery,
            Err(discovery_error) => {
                return Err(ModelFolderLookupError::DiscoveryFailed {
                    cause: discovery_error.to_string(),
                });
            }
        };
    let mut scanned_model_roots = vec![automatic_models_directory];
    scanned_model_roots.extend(configured_model_roots);
    let mut installed_model_ids = Vec::with_capacity(effective_discovery.discovered_models.len());
    let mut discovered_directories = Vec::with_capacity(installed_model_ids.capacity());
    let mut found_model = None;
    for discovered_model in effective_discovery.discovered_models {
        if discovered_model.model_id == model_id {
            found_model = Some((
                discovered_model.model_directory.clone(),
                discovered_model.model_family,
            ));
        }
        installed_model_ids.push(discovered_model.model_id);
        discovered_directories.push(discovered_model.model_directory);
    }
    let Some((model_directory, model_family)) = found_model else {
        return Err(ModelFolderLookupError::ModelNotFound {
            model_id: model_id.to_owned(),
        });
    };
    let nested_model_directories = nested_directories(&model_directory, &discovered_directories);
    Ok(ModelFolderLookup {
        model_directory,
        model_family,
        scanned_model_roots,
        installed_model_ids,
        nested_model_directories,
    })
}

/// The discovered model folders strictly inside `model_directory`, relative to it.
fn nested_directories(model_directory: &Path, discovered_directories: &[PathBuf]) -> Vec<PathBuf> {
    let mut nested_model_directories = Vec::new();
    for discovered_directory in discovered_directories {
        if let Ok(relative_directory) = discovered_directory.strip_prefix(model_directory)
            && !relative_directory.as_os_str().is_empty()
        {
            nested_model_directories.push(relative_directory.to_path_buf());
        }
    }
    nested_model_directories
}

/// The model folders the configuration lists; none when there is no file yet.
fn configured_model_roots(
    instance_paths: &AstronomicalInstancePaths,
) -> Result<Vec<PathBuf>, ModelFolderLookupError> {
    let config_file_path = instance_paths.config_file_path();
    let config_file_bytes = match std::fs::read(&config_file_path) {
        Ok(config_file_bytes) => config_file_bytes,
        Err(read_error) if read_error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(read_error) => {
            return Err(ModelFolderLookupError::ConfigUnreadable {
                config_file_path,
                cause: read_error.to_string(),
            });
        }
    };
    match AstronomicalConfig::load_v1_bytes(instance_paths.clone(), &config_file_bytes) {
        Ok(loaded_config) => Ok(loaded_config.model_directories().to_vec()),
        Err(config_error) => Err(ModelFolderLookupError::ConfigUnreadable {
            config_file_path,
            cause: config_error.to_string(),
        }),
    }
}

#[cfg(test)]
#[path = "model_folder_lookup_tests.rs"]
mod tests;
