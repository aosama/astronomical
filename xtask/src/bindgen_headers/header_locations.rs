//! Where provisioning reads archives and publishes headers, and where
//! verification finds a completed native build. The two environment
//! variables are honored because the native build orchestration and the
//! dependency bootstrap read the same ones; an explicit flag wins over both.

use std::path::{Path, PathBuf};

use astronomical_native_identity::probes::ToolchainOverrides;
use astronomical_native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};

use crate::bootstrap_native_dependencies::{
    NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE, resolve_cache_directory,
};

pub(crate) const HEADERS_CACHE_SUBDIRECTORY: &str = "bindgen-headers";
pub(crate) const EXTRACTION_TREE_MLX: &str = "mlx-src";
pub(crate) const EXTRACTION_TREE_MLX_C: &str = "mlx_c-src";
pub(crate) const COMPLETION_MARKER_FILE_NAME: &str = "complete";
const NATIVE_BUILD_STORE_VARIABLE: &str = "ASTRONOMICAL_NATIVE_BUILD_STORE_DIR";
const DEFAULT_NATIVE_BUILD_STORE_SUFFIX: &str = "Library/Caches/Astronomical/native-builds";
const NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH: &str =
    "crates/native-build-orchestration/src/native-build-store-schema-version";

pub(crate) fn native_dependency_cache_directory(
    explicit_directory: Option<&str>,
) -> Result<PathBuf, String> {
    let environment_directory = std::env::var(NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE).ok();
    let home_directory = std::env::var("HOME").ok();
    let chosen_directory = match explicit_directory {
        Some(explicit_directory) => Some(explicit_directory.to_owned()),
        None => environment_directory,
    };
    resolve_cache_directory(chosen_directory.as_deref(), home_directory.as_deref())
}

pub(crate) fn native_build_store_directory(
    explicit_directory: Option<&str>,
) -> Result<PathBuf, String> {
    let candidate_directory = match explicit_directory {
        Some(explicit_directory) => PathBuf::from(explicit_directory),
        None => match std::env::var(NATIVE_BUILD_STORE_VARIABLE) {
            Ok(environment_directory) if !environment_directory.is_empty() => {
                PathBuf::from(environment_directory)
            }
            _ => match std::env::var("HOME") {
                Ok(home_directory) => {
                    Path::new(&home_directory).join(DEFAULT_NATIVE_BUILD_STORE_SUFFIX)
                }
                Err(_) => {
                    return Err(format!(
                        "HOME is required when --native-build-store-dir and {NATIVE_BUILD_STORE_VARIABLE} are unset"
                    ));
                }
            },
        },
    };
    if !candidate_directory.is_absolute() {
        return Err(format!(
            "native build store directory must be an absolute path: {}",
            candidate_directory.display()
        ));
    }
    Ok(candidate_directory)
}

/// The shell pipeline hard-coded `v1`; reading the schema file keeps
/// verification pointed at the same store directory the build uses after a
/// schema bump.
pub(crate) fn native_build_store_schema_version(repository_root: &Path) -> Result<u32, String> {
    let schema_version_path = repository_root.join(NATIVE_BUILD_STORE_SCHEMA_VERSION_PATH);
    let schema_version_text = match std::fs::read_to_string(&schema_version_path) {
        Ok(schema_version_text) => schema_version_text,
        Err(read_error) => {
            return Err(format!(
                "native build store schema version is unavailable: {read_error}"
            ));
        }
    };
    match schema_version_text.trim().parse::<u32>() {
        Ok(schema_version) => Ok(schema_version),
        Err(_) => Err("native build store schema version must be an unsigned integer".to_owned()),
    }
}

/// Headers do not depend on the toolchain, so the extraction is keyed by the
/// source-only identity (pins, patches, patch pipeline, store schema): a
/// compiler update keeps a valid extraction, a pin or patch edit replaces it.
pub(crate) fn source_identity(repository_root: &Path) -> Result<String, String> {
    resolve_identity(
        repository_root,
        NativeBuildProfile::Core,
        IdentityScope::SourceOnly,
        None,
    )
}

/// Store entries are keyed by the full identity the build used, so
/// verification resolves profile and toolchain too.
pub(crate) fn built_identity(
    repository_root: &Path,
    native_build_profile: NativeBuildProfile,
    target_triple: &str,
) -> Result<String, String> {
    resolve_identity(
        repository_root,
        native_build_profile,
        IdentityScope::Full,
        Some(target_triple.to_owned()),
    )
}

fn resolve_identity(
    repository_root: &Path,
    native_build_profile: NativeBuildProfile,
    identity_scope: IdentityScope,
    target_triple: Option<String>,
) -> Result<String, String> {
    let fingerprint_request = FingerprintRequest {
        repository_root,
        profile: native_build_profile,
        scope: identity_scope,
        overrides: ToolchainOverrides {
            target_triple,
            ..ToolchainOverrides::none()
        },
    };
    match native_build_identity(&fingerprint_request) {
        Ok(native_identity) => Ok(native_identity),
        Err(identity_error) => Err(format!("native build identity failed: {identity_error}")),
    }
}

pub(crate) fn extraction_directory(cache_directory: &Path, source_identity: &str) -> PathBuf {
    cache_directory
        .join(HEADERS_CACHE_SUBDIRECTORY)
        .join(source_identity)
}

pub(crate) fn built_entry_directory(
    store_directory: &Path,
    schema_version: u32,
    built_identity: &str,
) -> PathBuf {
    store_directory
        .join(format!("v{schema_version}"))
        .join("entries")
        .join(built_identity)
}
