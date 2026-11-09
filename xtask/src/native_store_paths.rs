//! Where the machine-wide native build store keeps one identity's entry.
//!
//! xtask has no dependency on `astronomical-native-build-orchestration`, which
//! owns the store, so it mirrors the two facts it needs: the default root
//! under the user's cache directory, and the schema version file that names
//! the store's layout generation. Packaging reads the MLX metallib from an
//! entry, and CI restores and saves entries as GitHub Actions caches; both
//! must agree with the store byte for byte, so the layout lives in one place.

use std::path::{Path, PathBuf};

const NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH: &str =
    "crates/native-build-orchestration/src/native-build-store-schema-version";
const NATIVE_BUILD_STORE_RELATIVE_TO_HOME: &str = "Library/Caches/Astronomical/native-builds";

/// The directory holding one native identity's build products in the default
/// store, which is what every packaging and CI caller uses.
pub(crate) fn native_store_entry_directory(
    home_directory: &Path,
    schema_version: u32,
    native_identity: &str,
) -> PathBuf {
    home_directory
        .join(NATIVE_BUILD_STORE_RELATIVE_TO_HOME)
        .join(format!("v{schema_version}"))
        .join("entries")
        .join(native_identity)
}

pub(crate) fn native_store_schema_version(repository_root: &Path) -> Result<u32, String> {
    let schema_version_path = repository_root.join(NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH);
    let schema_version_text = match std::fs::read_to_string(&schema_version_path) {
        Ok(schema_version_text) => schema_version_text,
        Err(read_error) => {
            return Err(format!(
                "native build store schema version is unavailable: {read_error}"
            ));
        }
    };
    parse_schema_version(&schema_version_text)
}

pub(crate) fn parse_schema_version(schema_version_text: &str) -> Result<u32, String> {
    match schema_version_text.trim().parse::<u32>() {
        Ok(schema_version) => Ok(schema_version),
        Err(_) => Err("native build store schema version must be an unsigned integer".to_owned()),
    }
}
