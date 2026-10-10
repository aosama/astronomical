//! macOS unified-buffer-cache retention for weight file descriptors.
//!
//! Empirically pinned semantics of `fcntl(F_NOCACHE)` on APFS (issue #1120):
//! reads through a flagged descriptor that miss the cache are never inserted,
//! reads that hit already-cached pages are still served from the cache, the
//! flag is scoped to one open file description (never the vnode), and it dies
//! with the descriptor. A one-shot weight materialization therefore leaves no
//! file-cache ghost beside the wired MLX buffers, while descriptors of the
//! same files held by the expert pager keep their second-level cache.
//!
//! Why this matters to serving: the resident sweep cells measured gigabytes of
//! operating-system page-ins against ZERO bytes of our own positional reads.
//! The kernel, not the expert pager, was moving the weight bytes, and every
//! re-fault arrived unbatched. Keeping one-shot materialization out of the file
//! cache removes that second, invisible copy of the artifact from competing
//! with the wired working set.
//!
//! The load-bearing consequence is the descriptor rule below: `F_NOCACHE`
//! attaches to an open file description, so a cloned or inherited descriptor
//! silently inherits the policy of whoever flagged it first. Every caller that
//! requests `MaterializeOnce` must therefore own a FRESH descriptor.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::fd::AsRawFd;

use thiserror::Error;

/// How the unified buffer cache may retain one weight file's pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightsFileCacheRetention {
    /// The file is read exactly once while weights materialize into MLX
    /// buffers; cached insertion is disabled so no ghost copy remains beside
    /// the wired working set.
    MaterializeOnce,
    /// The file is a hot re-read source (expert paging); the kernel file cache
    /// acts as a second-level cache and must stay populated.
    ReuseAcrossReads,
}

/// A weights file cannot be opened under its cache-retention policy.
#[derive(Debug, Error)]
pub enum WeightsFileCacheRetentionError {
    #[error("opening weights file {source_path} for {retention:?} retention failed: {source}")]
    OpenFailed {
        source_path: PathBuf,
        retention: WeightsFileCacheRetention,
        source: io::Error,
    },
}

/// Opens a weight source file under an explicit cache-retention policy.
///
/// Resident materialization must open fresh descriptors rather than cloning
/// pager descriptors: a clone shares the open file description, so flagging it
/// would disable cache insertion for the pager's own reads on that file.
pub fn open_weights_file(
    source_path: &Path,
    retention: WeightsFileCacheRetention,
) -> Result<File, WeightsFileCacheRetentionError> {
    let weights_file =
        File::open(source_path).map_err(|source| WeightsFileCacheRetentionError::OpenFailed {
            source_path: source_path.to_owned(),
            retention,
            source,
        })?;
    apply_weights_file_cache_retention(&weights_file, source_path, retention);
    Ok(weights_file)
}

/// Applies a cache-retention policy to one exclusively owned descriptor.
///
/// The caller must fully own the open file description: inherited, cloned, or
/// shared descriptors would silently change caching behavior for every other
/// holder. `ReuseAcrossReads` is an explicit no-op so call sites name the
/// policy on both paths. A rejected no-insert flag (for example on a
/// filesystem without `F_NOCACHE` support) is a warning, not a load failure:
/// serving continues with default caching so an unsupported filesystem never
/// blocks model loading.
pub fn apply_weights_file_cache_retention(
    weights_file: &File,
    source_path: &Path,
    retention: WeightsFileCacheRetention,
) {
    if retention == WeightsFileCacheRetention::ReuseAcrossReads {
        return;
    }
    apply_no_cache_flag(weights_file, source_path);
}

#[cfg(target_os = "macos")]
fn apply_no_cache_flag(weights_file: &File, source_path: &Path) {
    // SAFETY: fcntl with F_NOCACHE takes ownership of the integer argument and
    // touches no memory owned by this call.
    let fcntl_status = unsafe { libc::fcntl(weights_file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if fcntl_status != 0 {
        let flag_error = io::Error::last_os_error();
        tracing::warn!(
            source_path = %source_path.display(),
            error = %flag_error,
            "the filesystem rejected the F_NOCACHE weight-materialization flag; continuing with default file caching"
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn apply_no_cache_flag(_weights_file: &File, _source_path: &Path) {
    // MLX serving targets Apple platforms; other platforms keep default cache
    // behavior rather than approximating F_NOCACHE with different semantics.
}
