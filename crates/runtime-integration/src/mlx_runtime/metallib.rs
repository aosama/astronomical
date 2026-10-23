//! Process-global selection, validation, and installation of the MLX AOT
//! (Ahead-Of-Time compiled) Metal library.
//!
//! # Why the Metallib path is pinned once, with proof
//!
//! MLX-C installs the Metallib path as process-global native state. If two
//! code paths installed different archives, whichever ran second would
//! silently redefine every Metal kernel for the whole process. The
//! `RUNTIME_METALLIB_PATH` static turns that hazard into a loud
//! [`MlxRuntimeError::MetallibAlreadyConfigured`].
//!
//! The install is also *verified*, not just chosen: the exact byte size and
//! SHA-256 digest of the build-produced archive are baked in as compile-time
//! environment values by the build script
//! (`build-orchestration/native_linking.rs`), and
//! [`validate_metallib_path`] re-checks every candidate before MLX ever sees
//! it. That closes two real failure modes: a relocated or repackaged app
//! bundle whose Resources tree was modified, and a symlink swap pointing the
//! bundled path at attacker-chosen bytes. The validation is the runtime half
//! of a closed loop — the build decided what the bytes should be, and the
//! runtime refuses anything else.

use std::{
    env,
    fs::OpenOptions,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::Mutex,
};

use sha2::{Digest, Sha256};

use crate::MlxRuntimeError;

use super::error_classification;

/// The once-per-process installed path. `None` until first install.
static RUNTIME_METALLIB_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

const MLX_METALLIB_PATH_ENVIRONMENT_VARIABLE: &str = "ASTRONOMICAL_MLX_METALLIB_PATH";
const EXPECTED_METALLIB_SHA256_HEX: &str = env!("ASTRONOMICAL_MLX_METALLIB_SHA256");
const EXPECTED_METALLIB_SIZE_BYTES_TEXT: &str = env!("ASTRONOMICAL_MLX_METALLIB_SIZE_BYTES");

/// Returns the native-store metallib produced at compile time.
///
/// Unpackaged `cargo run` falls back to this path when the executable is not
/// inside an app bundle. Shipped apps must load the copy packaged under
/// `Contents/Resources/share/mlx/mlx.metallib` because `~/Library/Caches` is
/// purgeable.
#[must_use]
pub fn compiled_metallib_path() -> &'static Path {
    Path::new(env!("ASTRONOMICAL_MLX_METALLIB_PATH"))
}

/// Verifies that a packaged or relocated metallib exactly matches the build-produced bytes.
///
/// Three checks, each with a reason:
///
/// 1. **Regular, non-symlink file** — inspected via `symlink_metadata` and,
///    on open, `O_NOFOLLOW`, so a link cannot redirect the read between the
///    check and the digest. A symlinked metallib is treated as "not the file
///    we built", not followed.
/// 2. **Exact size** — the cheap check first; a truncated or padded archive
///    fails here without hashing gigabytes.
/// 3. **Exact SHA-256 digest** — byte-for-byte equality with the
///    build-produced archive recorded at compile time.
///
/// # Errors
///
/// Returns [`MlxRuntimeError::InvalidMetallibPath`] describing the first
/// failed check.
pub fn validate_metallib_path(metallib_path: impl AsRef<Path>) -> Result<(), MlxRuntimeError> {
    let metallib_path = metallib_path.as_ref();
    let path_metadata = metallib_path.symlink_metadata().map_err(|source| {
        MlxRuntimeError::InvalidMetallibPath {
            description: format!("cannot inspect {metallib_path:?}: {source}"),
        }
    })?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!("path must name a regular non-symlink file: {metallib_path:?}"),
        });
    }
    let mut metallib_file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(metallib_path)
        .map_err(|source| MlxRuntimeError::InvalidMetallibPath {
            description: format!(
                "cannot open {metallib_path:?} without following symlinks: {source}"
            ),
        })?;
    let opened_metadata =
        metallib_file
            .metadata()
            .map_err(|source| MlxRuntimeError::InvalidMetallibPath {
                description: format!("cannot inspect opened metallib {metallib_path:?}: {source}"),
            })?;
    if !opened_metadata.is_file() {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!("opened metallib is not a regular file: {metallib_path:?}"),
        });
    }
    let expected_size_bytes = EXPECTED_METALLIB_SIZE_BYTES_TEXT
        .parse::<u64>()
        .map_err(|_| MlxRuntimeError::InvalidMetallibPath {
            description: "build-produced metallib size is invalid".to_owned(),
        })?;
    if opened_metadata.len() != expected_size_bytes {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!(
                "metallib size differs from the build-produced metallib: expected {expected_size_bytes}, got {}",
                opened_metadata.len()
            ),
        });
    }
    let actual_sha256_hex = sha256_reader_hex(&mut metallib_file).map_err(|source| {
        MlxRuntimeError::InvalidMetallibPath {
            description: format!("cannot digest metallib {metallib_path:?}: {source}"),
        }
    })?;
    if actual_sha256_hex != EXPECTED_METALLIB_SHA256_HEX {
        return Err(MlxRuntimeError::InvalidMetallibPath {
            description: format!(
                "metallib digest differs from the build-produced metallib: {metallib_path:?}"
            ),
        });
    }
    Ok(())
}

/// Resolves and validates the Metallib path this process should install.
///
/// Resolution order is owned by [`crate::mlx_metallib_path`] (environment
/// override, app bundle, compile-time fallback); this function adds the
/// validation gate so an invalid candidate fails here, before any native
/// state changes.
pub(crate) fn configured_metallib_path() -> Result<PathBuf, MlxRuntimeError> {
    let environment_override_path =
        env::var_os(MLX_METALLIB_PATH_ENVIRONMENT_VARIABLE).map(PathBuf::from);
    let current_executable_path =
        env::current_exe().map_err(|source| MlxRuntimeError::InvalidMetallibPath {
            description: format!("cannot resolve the current executable path: {source}"),
        })?;
    let metallib_path = crate::mlx_metallib_path::resolve_mlx_metallib_path(
        environment_override_path,
        &current_executable_path,
        Some(compiled_metallib_path()),
    )?;
    validate_metallib_path(&metallib_path)?;
    Ok(metallib_path)
}

/// Installs the validated path into MLX's process-global Metallib setting.
///
/// Idempotent for the *same* path (initialization and the device probe both
/// call this); a *different* path is rejected, because it would redefine
/// Metal kernels under a running process.
pub(crate) fn configure_metallib_path(metallib_path: &Path) -> Result<(), MlxRuntimeError> {
    let mut configured_path = error_classification::lock_unpoisoned(&RUNTIME_METALLIB_PATH);
    if let Some(existing_path) = configured_path.as_ref() {
        if existing_path != metallib_path {
            return Err(MlxRuntimeError::MetallibAlreadyConfigured {
                configured_path: existing_path.clone(),
            });
        }
        return Ok(());
    }
    // MLX-C stores the path as a C string; the two conversions below reject
    // the only byte sequences that cannot cross that boundary (interior NUL
    // bytes, non-UTF-8 paths) instead of truncating silently.
    let metallib_path_text =
        metallib_path
            .to_str()
            .ok_or_else(|| MlxRuntimeError::InvalidMetallibPath {
                description: format!("path is not valid UTF-8: {metallib_path:?}"),
            })?;
    let metallib_path_c_string = std::ffi::CString::new(metallib_path_text).map_err(|_| {
        MlxRuntimeError::InvalidMetallibPath {
            description: "path contains an interior NUL byte".to_owned(),
        }
    })?;
    mlx_c_rust::set_metallib_path(&metallib_path_c_string).map_err(MlxRuntimeError::from)?;
    *configured_path = Some(metallib_path.to_path_buf());
    Ok(())
}

/// Streams a file through SHA-256 without loading it into memory.
///
/// The Metallib is tens of megabytes; a fixed 64 KiB window keeps the digest
/// allocation flat regardless of archive size.
fn sha256_reader_hex(source_reader: &mut impl Read) -> std::io::Result<String> {
    let mut digest = Sha256::new();
    let mut digest_buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = source_reader.read(&mut digest_buffer)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&digest_buffer[..bytes_read]);
    }
    Ok(hex_digest_bytes(&digest.finalize()))
}

fn hex_digest_bytes(digest_bytes: &[u8]) -> String {
    digest_bytes
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect()
}
