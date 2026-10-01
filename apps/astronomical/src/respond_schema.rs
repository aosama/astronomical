//! Reads the `--schema` input for `astronomical respond`.
//!
//! The CLI owns only the file plumb: the schema file must be readable, valid
//! UTF-8, and within the shared schema byte bound. Schema validation itself
//! is the daemon's job, so a malformed schema is rejected at the daemon with
//! a reason naming the violated rule.

use std::path::Path;

use astronomical_ipc_protocol::MAXIMUM_CHAT_SCHEMA_JSON_BYTES;

use crate::errors::RespondError;

/// Reads the schema file into the raw JSON text the daemon request carries.
pub fn read_schema_input(schema_path: &Path) -> Result<String, RespondError> {
    let schema_bytes =
        std::fs::read(schema_path).map_err(|read_error| RespondError::SchemaReadFailed {
            path: schema_path.to_path_buf(),
            cause: read_error.to_string(),
        })?;
    if schema_bytes.len() > MAXIMUM_CHAT_SCHEMA_JSON_BYTES {
        return Err(RespondError::SchemaTooLarge {
            actual_bytes: schema_bytes.len(),
            maximum_bytes: MAXIMUM_CHAT_SCHEMA_JSON_BYTES,
        });
    }
    String::from_utf8(schema_bytes).map_err(|encoding_error| RespondError::SchemaNotUtf8 {
        path: schema_path.to_path_buf(),
        cause: encoding_error.utf8_error().to_string(),
    })
}
