//! Whole-document reads, saves, and resets for tools that edit `config.json`.
//!
//! Every write names the bytes it was based on, so a save cannot silently undo an
//! edit made elsewhere since the editor loaded the file: hand edits are a supported
//! way to change this file, and it has no other lock against them.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::AstronomicalConfigError;
use crate::config_document::UserConfigFile;
use crate::config_file::{
    CONFIG_SCHEMA_BYTES, parse_and_validate_v1, read_existing_config_file_bytes,
    write_adjacent_schema, write_config_file_bytes_atomically,
};
use crate::duplicate_key_json;

const CONFIG_FILE_NAME: &str = "config.json";
/// Reset keeps one previous document, because it discards every setting at once.
const PREVIOUS_CONFIG_FILE_NAME: &str = "config.previous.json";

/// The file as it stands now, even when a hand edit left it unreadable, so an
/// editor can still offer a reset instead of a dead end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigDocumentSnapshot {
    pub document: Option<serde_json::Value>,
    pub read_error: Option<String>,
    /// Identity of the exact bytes; `None` when no file exists yet.
    pub document_digest: Option<String>,
}

/// The bundled JSON Schema that describes every setting.
#[must_use]
pub fn config_schema_bytes() -> &'static [u8] {
    CONFIG_SCHEMA_BYTES
}

fn lowercase_hex_digest(bytes: &[u8]) -> String {
    astronomical_lowercase_hex::encode_lowercase_hex(&Sha256::digest(bytes))
}

/// Reads the document without migrating or rewriting anything.
pub fn read_config_document(
    state_directory: impl AsRef<Path>,
) -> Result<ConfigDocumentSnapshot, AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join(CONFIG_FILE_NAME);
    let Some(config_file_bytes) = read_existing_config_file_bytes(&config_file_path)? else {
        return Ok(ConfigDocumentSnapshot {
            document: None,
            read_error: None,
            document_digest: None,
        });
    };
    let document_digest = Some(lowercase_hex_digest(&config_file_bytes));
    match duplicate_key_json::parse_json_rejecting_duplicates(&config_file_path, &config_file_bytes)
    {
        Ok(document) => Ok(ConfigDocumentSnapshot {
            document: Some(document),
            read_error: None,
            document_digest,
        }),
        Err(parse_error) => Ok(ConfigDocumentSnapshot {
            document: None,
            read_error: Some(parse_error.to_string()),
            document_digest,
        }),
    }
}

/// Checks a candidate against the strict v1 contract without touching the file.
pub fn validate_config_document(
    state_directory: impl AsRef<Path>,
    candidate_document: &serde_json::Value,
) -> Result<(), AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join(CONFIG_FILE_NAME);
    parse_and_validate_v1(&config_file_path, candidate_document.clone()).map(drop)
}

fn require_expected_bytes(
    config_file_path: &Path,
    expected_document_digest: Option<&str>,
) -> Result<Option<Vec<u8>>, AstronomicalConfigError> {
    let current_bytes = read_existing_config_file_bytes(config_file_path)?;
    let current_digest = current_bytes.as_deref().map(lowercase_hex_digest);
    if current_digest.as_deref() != expected_document_digest {
        return Err(AstronomicalConfigError::ConfigChangedDuringUpdate);
    }
    Ok(current_bytes)
}

fn write_user_config_file(
    config_file_path: &Path,
    user_config_file: &UserConfigFile,
) -> Result<String, AstronomicalConfigError> {
    let config_file_bytes = match serde_json::to_vec_pretty(user_config_file) {
        Ok(config_file_bytes) => config_file_bytes,
        Err(source) => {
            return Err(AstronomicalConfigError::SerializeConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            });
        }
    };
    // The schema precedes the document so every committed config stays inspectable.
    write_adjacent_schema(config_file_path)?;
    write_config_file_bytes_atomically(config_file_path, &config_file_bytes)?;
    Ok(lowercase_hex_digest(&config_file_bytes))
}

/// Validates and atomically replaces the document, returning its new digest.
pub fn save_config_document(
    state_directory: impl AsRef<Path>,
    expected_document_digest: Option<&str>,
    candidate_document: &serde_json::Value,
) -> Result<String, AstronomicalConfigError> {
    let config_file_path = state_directory.as_ref().join(CONFIG_FILE_NAME);
    let user_config_file = parse_and_validate_v1(&config_file_path, candidate_document.clone())?;
    require_expected_bytes(&config_file_path, expected_document_digest)?;
    write_user_config_file(&config_file_path, &user_config_file)
}

/// Replaces the document with first-run defaults, keeping the old bytes beside it.
///
/// Works even when the current file is unreadable, which is when it matters most.
pub fn reset_config_document(
    state_directory: impl AsRef<Path>,
    expected_document_digest: Option<&str>,
) -> Result<String, AstronomicalConfigError> {
    let state_directory = state_directory.as_ref();
    let config_file_path = state_directory.join(CONFIG_FILE_NAME);
    let current_bytes = require_expected_bytes(&config_file_path, expected_document_digest)?;
    if let Some(current_bytes) = current_bytes {
        write_config_file_bytes_atomically(
            &state_directory.join(PREVIOUS_CONFIG_FILE_NAME),
            &current_bytes,
        )?;
    }
    write_user_config_file(&config_file_path, &UserConfigFile::minimal())
}

#[cfg(test)]
#[path = "config_document_edit_tests.rs"]
mod tests;
