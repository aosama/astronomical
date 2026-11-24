//! Reading and writing the artifact's JSON documents, with every failure
//! mapped to an error that names the file.

use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::QuantizationError;

/// Writes `document` as pretty JSON with a trailing newline, creating the
/// parent directory (the `optiq/` subfolder on first write).
pub fn write_pretty_json<Document: Serialize>(
    document: &Document,
    document_path: &Path,
) -> Result<(), QuantizationError> {
    let output_directory = document_path.parent().unwrap_or(document_path);
    if let Err(create_error) = std::fs::create_dir_all(output_directory) {
        return Err(write_failure(output_directory, &create_error.to_string()));
    }
    let mut document_text = match serde_json::to_string_pretty(document) {
        Ok(document_text) => document_text,
        Err(serialize_error) => {
            return Err(write_failure(
                output_directory,
                &serialize_error.to_string(),
            ));
        }
    };
    document_text.push('\n');
    match std::fs::write(document_path, document_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(write_failure(output_directory, &write_error.to_string())),
    }
}

/// Reads and parses a JSON document, returning the failure text for the
/// caller to wrap in its own error variant.
pub fn read_json<Document: DeserializeOwned>(document_path: &Path) -> Result<Document, String> {
    let document_text = match std::fs::read_to_string(document_path) {
        Ok(document_text) => document_text,
        Err(read_error) => return Err(read_error.to_string()),
    };
    match serde_json::from_str(&document_text) {
        Ok(document) => Ok(document),
        Err(parse_error) => Err(parse_error.to_string()),
    }
}

fn write_failure(output_directory: &Path, description: &str) -> QuantizationError {
    QuantizationError::ArtifactWriteFailure {
        output_directory: output_directory.to_path_buf(),
        description: description.to_owned(),
    }
}
