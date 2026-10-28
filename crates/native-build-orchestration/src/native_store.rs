//! Compatibility-keyed publication of the pinned MLX native runtime.
//!
//! # Why a store outside Cargo's `target/`
//!
//! Cargo package versions intentionally do not own this store. The
//! `mlx-c-rust` build script regenerates Rust bindings in its own `OUT_DIR`,
//! while expensive native products remain reusable until an actual native
//! toolchain input changes — a Rust-only edit must never rebuild MLX, and a
//! new checkout or worktree on the same machine must reuse the same warm
//! entry.
//!
//! # Store layout
//!
//! ```text
//! <root>/v<schema>/entries/<64-hex identity>/   published payloads
//! <root>/v<schema>/locks/<identity>.lock        exclusive per-identity locks
//! <root>/v<schema>/staging/<id>.<pid>.<nonce>/  in-progress builds
//! ```
//!
//! The schema directory version (`native-build-store-schema-version`, a
//! fingerprinted input) fences every entry whenever the layout or validation
//! semantics change: bump it and old entries are simply never looked at
//! again.
//!
//! # The publication protocol
//!
//! Reuse is validated, never trusted: an entry counts only if it is a real
//! directory whose completion marker matches its identity, whose SHA-256
//! manifest matches the payload byte-for-byte, and whose profile's products
//! are all present. A failing entry is removed and rebuilt. A fresh build
//! happens in a staging directory, is fully synced to disk, is validated,
//! and only then atomically becomes the entry via rename — an interrupted
//! build can therefore never leave a half-published entry that a later run
//! would mistake for reusable.
//!
//! Concurrency is per-identity: one builder holds the lock file (with
//! progress heartbeats while waiting), other identities build in parallel.

use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use fs4::TryLockError;

mod manifest;
#[cfg(test)]
mod tests;

use manifest::{
    copy_directory_without_symlinks, copy_required_file, sync_directory, sync_directory_tree,
    validate_entry, write_manifest, write_synced_file,
};

/// Bumped whenever the store layout or validation semantics change; see the
/// module documentation. Content is an unsigned integer in the file beside
/// this module.
const STORE_SCHEMA_VERSION_TEXT: &str = include_str!("native-build-store-schema-version");
const LOCK_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// Which optional native products a profile builds and publishes.
///
/// `core` is the serving product; `core+memory-contract` adds the explicit
/// allocator memory-contract probe binary used by the acceptance lanes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeBuildProfile {
    should_build_memory_contract_probe: bool,
}

impl NativeBuildProfile {
    pub const fn core() -> Self {
        Self {
            should_build_memory_contract_probe: false,
        }
    }

    pub const fn new(should_build_memory_contract_probe: bool) -> Self {
        Self {
            should_build_memory_contract_probe,
        }
    }

    pub const fn should_build_memory_contract_probe(self) -> bool {
        self.should_build_memory_contract_probe
    }

    /// The identity-library spelling of this profile
    /// (`core` or `core+memory-contract`); shared with
    /// `astronomical-native-identity` so both sides agree on the manifest
    /// and command-line vocabulary.
    pub fn identity_name(self) -> &'static str {
        if self.should_build_memory_contract_probe {
            "core+memory-contract"
        } else {
            "core"
        }
    }

    // The standalone native-build tool resolves profiles from CLI identity
    // names; the Cargo build script only ever constructs profiles directly,
    // so this parser is dead code from the build script's point of view.
    #[allow(dead_code)]
    pub fn from_identity_name(identity_name: &str) -> Option<Self> {
        match identity_name {
            "core" => Some(Self::core()),
            "core+memory-contract" => Some(Self::new(true)),
            _ => None,
        }
    }
}

/// One identity-keyed view of the machine-wide native build store.
pub struct NativeBuildStore {
    root_directory: PathBuf,
    native_identity: String,
    native_build_profile: NativeBuildProfile,
}

impl NativeBuildStore {
    /// Validates the store root, schema version, and identity spelling.
    ///
    /// # Errors
    ///
    /// Rejects relative roots (the store is machine-global and must be
    /// unambiguous), malformed schema versions, and identities that are not
    /// exactly 64 lowercase hexadecimal characters — the digest format the
    /// identity library produces; anything else would create entry
    /// directories that no later run could find again.
    pub fn new(
        root_directory: &Path,
        native_identity: &str,
        native_build_profile: NativeBuildProfile,
    ) -> Result<Self, Box<dyn Error>> {
        if !root_directory.is_absolute() {
            return Err(format!(
                "native build store must be an absolute path: {}",
                root_directory.display()
            )
            .into());
        }
        validate_store_schema_version()?;
        validate_native_identity(native_identity)?;
        Ok(Self {
            root_directory: root_directory.to_owned(),
            native_identity: native_identity.to_owned(),
            native_build_profile,
        })
    }

    /// The published entry directory for this store's identity.
    pub fn entry_directory(&self) -> PathBuf {
        self.schema_directory()
            .join("entries")
            .join(&self.native_identity)
    }

    /// Reuses the published entry for this identity, or builds and publishes
    /// one under the exclusive identity lock.
    ///
    /// `native_builder` receives the CMake build directory to populate; its
    /// return value decides publication (the builder is the same shared
    /// `native_compile::build_pinned_native_runtime` in every caller).
    /// Status lines (`reused`/`building`/`published`/`corrupt`) land on
    /// stderr for build logs.
    pub fn resolve_or_build<F>(
        &self,
        native_builder: F,
    ) -> Result<NativeBuildArtifacts, Box<dyn Error>>
    where
        F: FnOnce(&Path) -> Result<(), Box<dyn Error>>,
    {
        self.create_store_directories()?;
        let _identity_lock = self.acquire_identity_lock()?;
        let entry_directory = self.entry_directory();
        match fs::symlink_metadata(&entry_directory) {
            Ok(_) => match validate_entry(
                &entry_directory,
                &self.native_identity,
                self.native_build_profile,
            ) {
                Ok(()) => {
                    eprintln!(
                        "[native-build-store] status=reused identity={}",
                        self.native_identity
                    );
                    return Ok(NativeBuildArtifacts::new(
                        entry_directory,
                        self.native_build_profile,
                        false,
                    ));
                }
                Err(validation_error) => {
                    eprintln!(
                        "[native-build-store] status=corrupt identity={} reason={validation_error}",
                        self.native_identity
                    );
                    remove_entry_without_following_symlinks(&entry_directory)?;
                }
            },
            Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {}
            Err(metadata_error) => return Err(metadata_error.into()),
        }

        let staging_directory = self.create_staging_directory()?;
        let cmake_build_directory = staging_directory.join("cmake");
        fs::create_dir(&cmake_build_directory)?;
        eprintln!(
            "[native-build-store] status=building identity={} profile={}",
            self.native_identity,
            self.native_build_profile.identity_name()
        );
        if let Err(build_error) = native_builder(&cmake_build_directory) {
            remove_staging_directory(&staging_directory);
            return Err(build_error);
        }
        let publication_result = self.publish_entry(&staging_directory, &cmake_build_directory);
        remove_staging_directory(&staging_directory);
        publication_result?;
        eprintln!(
            "[native-build-store] status=published identity={}",
            self.native_identity
        );
        Ok(NativeBuildArtifacts::new(
            entry_directory,
            self.native_build_profile,
            true,
        ))
    }

    fn schema_directory(&self) -> PathBuf {
        self.root_directory
            .join(format!("v{}", store_schema_version()))
    }

    fn create_store_directories(&self) -> Result<(), Box<dyn Error>> {
        for directory_name in ["entries", "locks", "staging"] {
            fs::create_dir_all(self.schema_directory().join(directory_name))?;
        }
        Ok(())
    }

    /// Takes the per-identity exclusive lock, waiting with progress logs.
    ///
    /// The lock file is created without truncation: it carries no content,
    /// but truncating while another process holds the lock would race that
    /// holder's view of the file.
    fn acquire_identity_lock(&self) -> Result<File, Box<dyn Error>> {
        let lock_file_path = self
            .schema_directory()
            .join("locks")
            .join(format!("{}.lock", self.native_identity));
        let lock_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_file_path)?;
        let wait_started_at = Instant::now();
        let mut next_progress_at = LOCK_PROGRESS_INTERVAL;
        loop {
            match fs4::FileExt::try_lock(&lock_file) {
                Ok(()) => return Ok(lock_file),
                Err(TryLockError::WouldBlock) => {
                    if wait_started_at.elapsed() >= next_progress_at {
                        eprintln!(
                            "[native-build-store] status=waiting identity={} elapsed_seconds={}",
                            self.native_identity,
                            wait_started_at.elapsed().as_secs()
                        );
                        next_progress_at += LOCK_PROGRESS_INTERVAL;
                    }
                    thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(TryLockError::Error(lock_error)) => return Err(lock_error.into()),
            }
        }
    }

    /// Unique per-invocation staging directory (identity, process id, nonce).
    fn create_staging_directory(&self) -> Result<PathBuf, Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let staging_directory = self.schema_directory().join("staging").join(format!(
            "{}.{}.{}",
            self.native_identity,
            std::process::id(),
            nonce
        ));
        fs::create_dir(&staging_directory)?;
        Ok(staging_directory)
    }

    /// Copies the reusable products out of the CMake tree, syncs them to
    /// disk, writes the integrity manifest and completion marker, and
    /// atomically renames the payload into place as the entry.
    ///
    /// Symlinks are rejected everywhere (see the manifest module) so a
    /// compromised cache cannot smuggle products from elsewhere on the
    /// machine into a build input.
    fn publish_entry(
        &self,
        staging_directory: &Path,
        cmake_build_directory: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let payload_directory = staging_directory.join("payload");
        fs::create_dir(&payload_directory)?;
        copy_directory_without_symlinks(
            &cmake_build_directory.join("_deps/mlx_c-src/mlx"),
            &payload_directory.join("include/mlx"),
        )?;
        copy_required_file(
            &cmake_build_directory.join("lib/libmlx.a"),
            &payload_directory.join("lib/libmlx.a"),
        )?;
        copy_required_file(
            &cmake_build_directory.join("lib/libmlxc.a"),
            &payload_directory.join("lib/libmlxc.a"),
        )?;
        copy_required_file(
            &cmake_build_directory.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
            &payload_directory.join("share/mlx/mlx.metallib"),
        )?;
        if self
            .native_build_profile
            .should_build_memory_contract_probe()
        {
            copy_required_file(
                &cmake_build_directory.join("bin/mlx_memory_contract_probe"),
                &payload_directory.join("bin/mlx_memory_contract_probe"),
            )?;
        }
        // Durability before visibility: everything is synced before the
        // rename makes the entry discoverable, so a crash cannot publish an
        // entry whose bytes never reached the disk.
        sync_directory_tree(&payload_directory)?;
        write_manifest(
            &payload_directory,
            &self.native_identity,
            self.native_build_profile,
        )?;
        write_synced_file(
            &payload_directory.join("complete"),
            format!("{}\n", self.native_identity).as_bytes(),
        )?;
        sync_directory(&payload_directory)?;
        let entry_directory = self.entry_directory();
        fs::rename(&payload_directory, &entry_directory)?;
        sync_directory(
            entry_directory
                .parent()
                .ok_or("native entry directory has no parent")?,
        )?;
        // Validate what was published, not what was staged: the entry is
        // what the next run will consume.
        validate_entry(
            &entry_directory,
            &self.native_identity,
            self.native_build_profile,
        )
    }
}

fn store_schema_version() -> &'static str {
    STORE_SCHEMA_VERSION_TEXT.trim()
}

fn validate_store_schema_version() -> Result<(), Box<dyn Error>> {
    let schema_version = store_schema_version();
    if schema_version.is_empty()
        || schema_version
            .bytes()
            .any(|version_byte| !version_byte.is_ascii_digit())
    {
        return Err("native build store schema version must be an unsigned integer".into());
    }
    Ok(())
}

/// A published entry as consumed by linking and status reporting.
#[derive(Debug)]
pub struct NativeBuildArtifacts {
    entry_directory: PathBuf,
    native_build_profile: NativeBuildProfile,
    was_built: bool,
}

impl NativeBuildArtifacts {
    fn new(
        entry_directory: PathBuf,
        native_build_profile: NativeBuildProfile,
        was_built: bool,
    ) -> Self {
        Self {
            entry_directory,
            native_build_profile,
            was_built,
        }
    }

    /// Whether this run produced the entry (`true`) or reused it (`false`);
    /// the CI status file reports exactly this bit.
    pub const fn was_built(&self) -> bool {
        self.was_built
    }

    pub fn native_library_directory(&self) -> PathBuf {
        self.entry_directory.join("lib")
    }

    pub fn mlx_library_path(&self) -> PathBuf {
        self.native_library_directory().join("libmlx.a")
    }

    pub fn metallib_path(&self) -> PathBuf {
        self.entry_directory.join("share/mlx/mlx.metallib")
    }

    /// Present only for the `core+memory-contract` profile.
    pub fn memory_contract_probe_path(&self) -> Option<PathBuf> {
        self.native_build_profile
            .should_build_memory_contract_probe()
            .then(|| self.entry_directory.join("bin/mlx_memory_contract_probe"))
    }
}

fn validate_native_identity(native_identity: &str) -> Result<(), Box<dyn Error>> {
    if native_identity.len() != 64
        || native_identity.bytes().any(|identity_byte| {
            !identity_byte.is_ascii_hexdigit() || identity_byte.is_ascii_uppercase()
        })
    {
        return Err("native build identity must be 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

/// Staging cleanup is best-effort: a failed build must still report its real
/// error, so cleanup failures are logged, never propagated.
fn remove_staging_directory(staging_directory: &Path) {
    if let Err(cleanup_error) = fs::remove_dir_all(staging_directory) {
        eprintln!(
            "[native-build-store] status=cleanup-error path={} reason={cleanup_error}",
            staging_directory.display()
        );
    }
}

/// Removes a corrupt entry without following symlinks.
///
/// If an attacker or accident left a *symlink* at the entry path, it is
/// unlinked as a file — never followed — so "cleanup" cannot delete whatever
/// the link points at.
fn remove_entry_without_following_symlinks(entry_path: &Path) -> Result<(), Box<dyn Error>> {
    let entry_metadata = fs::symlink_metadata(entry_path)?;
    if entry_metadata.is_dir() && !entry_metadata.file_type().is_symlink() {
        fs::remove_dir_all(entry_path)?;
    } else {
        fs::remove_file(entry_path)?;
    }
    Ok(())
}
