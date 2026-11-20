//! Derives a privacy-safe identity from one validated semantic configuration document.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::AstronomicalConfigError;
use crate::config_document::UserConfigFile;

pub(crate) fn configuration_generation(
    config_file_path: &Path,
    user_config_file: &UserConfigFile,
) -> Result<String, AstronomicalConfigError> {
    // The menu_bar section only changes what the menu-bar agent draws; no worker reads it. A
    // reload that sees a new generation and no field it recognizes replaces the worker, which
    // unloads the person's model, so the section is left out of the identity: writing it, even
    // an explicit default, must never cost a model load.
    let worker_relevant_config_file = UserConfigFile {
        menu_bar: None,
        ..user_config_file.clone()
    };
    // Hash the typed document so JSON whitespace and member order cannot create a false reload.
    let canonical_config_bytes =
        serde_json::to_vec(&worker_relevant_config_file).map_err(|source| {
            AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.to_path_buf(),
                source,
            }
        })?;
    let digest = Sha256::digest(canonical_config_bytes);
    Ok(lowercase_hex::encode_lowercase_hex(&digest))
}
