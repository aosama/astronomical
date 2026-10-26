//! Unit tests for `crate::verbs::respond::schema`: reading the `--schema` file from
//! disk with the shared byte and encoding bounds applied before any daemon
//! contact.

use std::path::PathBuf;

use astronomical_ipc_protocol::MAXIMUM_CHAT_SCHEMA_JSON_BYTES;

use crate::verbs::respond::RespondError;
use crate::verbs::respond::schema as respond_schema;

#[test]
fn should_reject_an_oversize_schema_file() {
    let schema_directory = tempfile::tempdir().expect("the schema directory should be creatable");
    let schema_path: PathBuf = schema_directory.path().join("oversize-schema.json");
    let oversize_schema_text = format!(
        "{{\"values\":[\"{}\"]}}",
        "x".repeat(MAXIMUM_CHAT_SCHEMA_JSON_BYTES)
    );
    std::fs::write(&schema_path, oversize_schema_text)
        .expect("the oversize schema file should be writable");

    match respond_schema::read_schema_input(&schema_path) {
        Err(RespondError::SchemaTooLarge {
            actual_bytes,
            maximum_bytes,
        }) => {
            assert!(actual_bytes > MAXIMUM_CHAT_SCHEMA_JSON_BYTES);
            assert_eq!(maximum_bytes, MAXIMUM_CHAT_SCHEMA_JSON_BYTES);
        }
        other => panic!("expected a schema size failure, got {other:?}"),
    }
}

#[test]
fn should_reject_a_non_utf8_schema_file() {
    let schema_directory = tempfile::tempdir().expect("the schema directory should be creatable");
    let schema_path: PathBuf = schema_directory.path().join("binary-schema.json");
    std::fs::write(&schema_path, [0xFF, 0xFE, 0xFC])
        .expect("the binary schema file should be writable");

    match respond_schema::read_schema_input(&schema_path) {
        Err(RespondError::SchemaNotUtf8 { path, .. }) => assert_eq!(path, schema_path),
        other => panic!("expected a UTF-8 failure, got {other:?}"),
    }
}
