//! `cargo xtask bootstrap-native-dependencies` — downloads and SHA-256-verifies
//! Astronomical's pinned MLX-family source archives into the native dependency
//! cache. The pins stay in the CMake manifest
//! (`third-party/native-dependency-manifest.cmake`) and are read through a thin
//! `cmake -P` edge; the manifest field policy lives in
//! `native_dependency_manifest.rs`. The pure policy (the verify-or-download
//! decision, free-space arithmetic, downloaded-size bounds) is unit-tested
//! beside this source; `curl`, `cmake`, and `df` stay thin edges (issue
//! #1250). The retired shell tool capped writers with `ulimit -f`; this port
//! drops that deliberately — `curl --max-filesize` plus the post-download
//! size check own the archive bound, and a filesystem block limit would only
//! turn an oversized download into an opaque SIGXFSZ failure.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::native_dependency_manifest::{ManifestEntry, parse_manifest_text};
use sha2::Digest;

const MAX_ARCHIVE_SIZE_BYTES: u64 = 268_435_456;
const MINIMUM_FREE_SPACE_BYTES: u64 = 2_147_483_648;
const NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE: &str = "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR";
const DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX: &str =
    "Library/Caches/Astronomical/native-dependencies";
const USAGE_TEXT: &str =
    "usage: cargo xtask bootstrap-native-dependencies [--cache-dir ABSOLUTE_PATH] [--verify]";

pub(crate) struct BootstrapOptions {
    pub(crate) cache_directory: Option<String>,
    pub(crate) is_verify_only: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProvisionDecision {
    VerifiedCached,
    Download,
}

/// A temp file that removes itself on every exit path, replacing the shell
/// tool's `trap cleanup 0`.
struct TemporaryFileGuard {
    path: PathBuf,
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl TemporaryFileGuard {
    /// Releases the guard after the temp path was consumed by an atomic
    /// rename, so the Drop cleanup no longer targets a moved file.
    fn defuse(self) {}
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let started_at = std::time::Instant::now();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{USAGE_TEXT}");
        println!();
        println!("Downloads and SHA-256-verifies Astronomical's pinned MLX source archives.");
        println!(
            "Default native dependency cache: $HOME/{DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX}"
        );
        println!();
        println!(
            "--cache-dir ABSOLUTE_PATH  Store or verify archives in this native dependency cache directory."
        );
        println!("--verify                   Verify cached archives without downloading.");
        return Ok(());
    }
    let options = parse_arguments(arguments)?;
    let environment_cache_directory = std::env::var(NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE)
        .ok()
        .filter(|directory| !directory.is_empty());
    let cache_directory = resolve_cache_directory(
        options
            .cache_directory
            .as_deref()
            .or(environment_cache_directory.as_deref()),
        std::env::var("HOME").ok().as_deref(),
    )?;

    log_step("validate-inputs", "start");
    std::fs::create_dir_all(&cache_directory)
        .map_err(|error| format!("native dependency cache directory failed: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cache_directory, std::fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("chmod failed: {error}"))?;
    }
    if !options.is_verify_only {
        let df_output = read_free_space_statistics(&cache_directory)?;
        let available_kib_blocks = parse_available_kib_blocks(&df_output)?;
        if !has_required_free_space(available_kib_blocks) {
            return Err(format!(
                "native dependency cache filesystem needs at least {MINIMUM_FREE_SPACE_BYTES} free bytes before downloading"
            ));
        }
    }
    log_step("validate-inputs", "success");

    log_step("read-pinned-manifest", "start");
    let manifest_text = read_pinned_manifest_text(repository_root)?;
    let manifest_entries = parse_manifest_text(&manifest_text)?;
    log_step("read-pinned-manifest", "success");

    log_step("verify-or-download-archives", "start");
    for manifest_entry in &manifest_entries {
        provision_archive(&cache_directory, manifest_entry, options.is_verify_only)?;
    }
    log_step("verify-or-download-archives", "success");

    println!(
        "[native-dependencies] native_dependency_cache_directory={} status=ready elapsed_seconds={}",
        cache_directory.display(),
        started_at.elapsed().as_secs()
    );
    Ok(())
}

pub(crate) fn parse_arguments(arguments: &[String]) -> Result<BootstrapOptions, String> {
    let mut cache_directory: Option<String> = None;
    let mut is_verify_only = false;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        match arguments[argument_index].as_str() {
            "--cache-dir" => {
                let Some(directory_value) = arguments.get(argument_index + 1) else {
                    return Err("--cache-dir requires an absolute path".to_owned());
                };
                cache_directory = Some(directory_value.clone());
                argument_index += 2;
            }
            "--verify" => {
                is_verify_only = true;
                argument_index += 1;
            }
            unrecognized_argument => {
                return Err(format!(
                    "unrecognized argument: {unrecognized_argument}\n{USAGE_TEXT}"
                ));
            }
        }
    }
    Ok(BootstrapOptions {
        cache_directory,
        is_verify_only,
    })
}

pub(crate) fn resolve_cache_directory(
    explicit_directory: Option<&str>,
    home_directory: Option<&str>,
) -> Result<PathBuf, String> {
    let candidate_directory = match explicit_directory {
        Some(directory) => PathBuf::from(directory),
        None => {
            let Some(home_directory) = home_directory else {
                return Err(format!(
                    "HOME is required when --cache-dir and {NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE} are unset"
                ));
            };
            PathBuf::from(home_directory).join(DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX)
        }
    };
    if !candidate_directory.is_absolute() {
        return Err(format!(
            "native dependency cache directory must be an absolute path: {}",
            candidate_directory.display()
        ));
    }
    Ok(candidate_directory)
}

/// Parses the available-blocks column of `df -k -P` output (POSIX format:
/// `Filesystem 1K-blocks Used Available Capacity Mounted on`), which macOS
/// and Linux both emit identically in the fourth column.
pub(crate) fn parse_available_kib_blocks(df_output: &str) -> Result<u64, String> {
    let data_line = df_output.lines().nth(1).ok_or_else(|| {
        "could not parse native dependency cache filesystem statistics".to_owned()
    })?;
    let available_block_text = data_line.split_whitespace().nth(3).ok_or_else(|| {
        "could not determine available native dependency cache filesystem blocks".to_owned()
    })?;
    available_block_text.parse::<u64>().map_err(|_| {
        "could not determine available native dependency cache filesystem blocks".to_owned()
    })
}

pub(crate) fn has_required_free_space(available_kib_blocks: u64) -> bool {
    available_kib_blocks.saturating_mul(1_024) >= MINIMUM_FREE_SPACE_BYTES
}

pub(crate) fn validate_downloaded_size(
    downloaded_size_bytes: u64,
    dependency_description: &str,
) -> Result<(), String> {
    if downloaded_size_bytes > MAX_ARCHIVE_SIZE_BYTES {
        return Err(format!(
            "{dependency_description} exceeds the {MAX_ARCHIVE_SIZE_BYTES}-byte archive limit"
        ));
    }
    Ok(())
}

pub(crate) fn decide_provision(
    cache_entry_exists: bool,
    cache_entry_is_regular_file: bool,
    cached_digest_matches: bool,
    is_verify_only: bool,
    dependency_description: &str,
    archive_path: &Path,
) -> Result<ProvisionDecision, String> {
    if cache_entry_exists && !cache_entry_is_regular_file {
        return Err(format!(
            "native dependency cache path is not a regular file: {}",
            archive_path.display()
        ));
    }
    if cache_entry_exists && cached_digest_matches {
        return Ok(ProvisionDecision::VerifiedCached);
    }
    if is_verify_only {
        return Err(format!(
            "missing or invalid {dependency_description} archive: {}",
            archive_path.display()
        ));
    }
    Ok(ProvisionDecision::Download)
}

/// `symlink_metadata` refuses symlinked cache entries: `is_file()` is false
/// for a link itself, so a planted symlink can never pass as the archive.
fn cache_entry_regular_file_state(archive_path: &Path) -> (bool, bool) {
    match std::fs::symlink_metadata(archive_path) {
        Ok(metadata) => (true, metadata.is_file()),
        Err(_) => (false, false),
    }
}

/// Streams the file in 64 KiB chunks through SHA-256: pinned source archives
/// can approach the 256 MiB bound, and a whole-file read would allocate that
/// per archive on every verify.
fn file_digest_matches(file_path: &Path, expected_sha256: &str) -> bool {
    let Ok(mut archive_file) = std::fs::File::open(file_path) else {
        return false;
    };
    let mut digest_hasher = sha2::Sha256::new();
    let mut read_chunk = [0u8; 64 * 1024];
    loop {
        match std::io::Read::read(&mut archive_file, &mut read_chunk) {
            Ok(0) => break,
            Ok(chunk_bytes) => digest_hasher.update(&read_chunk[..chunk_bytes]),
            Err(_) => return false,
        }
    }
    let digest_hex: String = digest_hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    digest_hex == expected_sha256
}

fn provision_archive(
    cache_directory: &Path,
    manifest_entry: &ManifestEntry,
    is_verify_only: bool,
) -> Result<(), String> {
    let archive_path = cache_directory.join(&manifest_entry.archive_file_name);
    let (cache_entry_exists, cache_entry_is_regular_file) =
        cache_entry_regular_file_state(&archive_path);
    let digest_matches = cache_entry_is_regular_file
        && file_digest_matches(&archive_path, &manifest_entry.archive_sha256);
    match decide_provision(
        cache_entry_exists,
        cache_entry_is_regular_file,
        digest_matches,
        is_verify_only,
        &manifest_entry.dependency_description,
        &archive_path,
    )? {
        ProvisionDecision::VerifiedCached => {
            println!(
                "[native-dependencies] archive={} status=verified-cached",
                manifest_entry.dependency_description
            );
        }
        ProvisionDecision::Download => {
            download_and_install_archive(cache_directory, manifest_entry, &archive_path)?;
        }
    }
    Ok(())
}

fn download_and_install_archive(
    cache_directory: &Path,
    manifest_entry: &ManifestEntry,
    archive_path: &Path,
) -> Result<(), String> {
    let download_destination = cache_directory.join(format!(
        ".{}.download.{}",
        manifest_entry.archive_file_name,
        unique_suffix()
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&download_destination)
        .map_err(|error| format!("download staging file failed: {error}"))?;
    let download_guard = TemporaryFileGuard {
        path: download_destination.clone(),
    };
    download_with_curl(&download_destination, &manifest_entry.archive_url).map_err(|error| {
        format!(
            "failed to download {} from {}: {error}",
            manifest_entry.dependency_description, manifest_entry.archive_url
        )
    })?;
    let downloaded_size_bytes = std::fs::metadata(&download_destination)
        .map(|metadata| metadata.len())
        .map_err(|error| format!("could not determine downloaded size: {error}"))?;
    validate_downloaded_size(
        downloaded_size_bytes,
        &manifest_entry.dependency_description,
    )?;
    if !file_digest_matches(&download_destination, &manifest_entry.archive_sha256) {
        return Err(format!(
            "{} SHA-256 verification failed",
            manifest_entry.dependency_description
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            &download_destination,
            std::fs::Permissions::from_mode(0o600),
        )
        .map_err(|error| format!("chmod failed: {error}"))?;
    }
    // Same-filesystem rename: the staging file lives inside the cache
    // directory, so the move into place is atomic like the shell tool's
    // `mv -f` was.
    std::fs::rename(&download_destination, archive_path)
        .map_err(|error| format!("move into place failed: {error}"))?;
    download_guard.defuse();
    println!(
        "[native-dependencies] archive={} status=downloaded-and-verified bytes={downloaded_size_bytes}",
        manifest_entry.dependency_description
    );
    Ok(())
}

fn download_with_curl(download_destination: &Path, download_url: &str) -> Result<(), String> {
    let curl_status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--tlsv1.2",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "--max-filesize",
        ])
        .arg(MAX_ARCHIVE_SIZE_BYTES.to_string())
        .arg("--output")
        .arg(download_destination)
        .arg(download_url)
        .status()
        .map_err(|error| format!("curl failed to start: {error}"))?;
    if curl_status.success() {
        Ok(())
    } else {
        Err(format!("curl exited unsuccessfully for {download_url}"))
    }
}

/// Reads the pinned manifest by delegating to the CMake script that owns the
/// pins; the manifest content never lives in two places.
fn read_pinned_manifest_text(repository_root: &Path) -> Result<String, String> {
    let manifest_cmake_path = repository_root.join("third-party/native-dependency-manifest.cmake");
    let temporary_manifest_path = std::env::temp_dir().join(format!(
        "astronomical-native-dependencies.{}.manifest",
        unique_suffix()
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_manifest_path)
        .map_err(|error| format!("temporary manifest file failed: {error}"))?;
    let _manifest_guard = TemporaryFileGuard {
        path: temporary_manifest_path.clone(),
    };
    let cmake_status = Command::new("cmake")
        .arg(format!(
            "-DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH={}",
            temporary_manifest_path.display()
        ))
        .arg("-P")
        .arg(&manifest_cmake_path)
        .status()
        .map_err(|error| format!("cmake failed to start: {error}"))?;
    if !cmake_status.success() {
        return Err(format!(
            "cmake manifest read failed for {}",
            manifest_cmake_path.display()
        ));
    }
    std::fs::read_to_string(&temporary_manifest_path)
        .map_err(|error| format!("native dependency manifest is unreadable: {error}"))
}

fn read_free_space_statistics(cache_directory: &Path) -> Result<String, String> {
    let df_output = Command::new("df")
        .args(["-k", "-P"])
        .arg(cache_directory)
        .output()
        .map_err(|error| format!("df failed to start: {error}"))?;
    if !df_output.status.success() {
        return Err("df could not stat the native dependency cache filesystem".to_owned());
    }
    Ok(String::from_utf8_lossy(&df_output.stdout).to_string())
}

/// Pid plus nanos: a crashed run can leave a staging file behind, and pid
/// recycling alone would make the next run's `create_new` collide with it.
fn unique_suffix() -> String {
    format!(
        "{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0)
    )
}

fn log_step(step_name: &str, step_status: &str) {
    eprintln!("[native-dependencies] step={step_name} status={step_status}");
}
