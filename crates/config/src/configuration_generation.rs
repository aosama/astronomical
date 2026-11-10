//! Derives a privacy-safe identity from one validated semantic configuration document.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::AstronomicalConfigError;
use crate::config_document::UserConfigFile;

pub(crate) fn configuration_generation(
    config_file_path: &Path,
    user_config_file: &UserConfigFile,
) -> Result<String, AstronomicalConfigError> {
    // Hash the typed document so JSON whitespace and member order cannot create a false reload.
    let canonical_config_bytes = serde_json::to_vec(user_config_file).map_err(|source| {
        AstronomicalConfigError::SerializeConfigFile {
            config_file_path: config_file_path.to_path_buf(),
            source,
        }
    })?;
    let digest = Sha256::digest(canonical_config_bytes);
    Ok(astronomical_lowercase_hex::encode_lowercase_hex(&digest))
}
