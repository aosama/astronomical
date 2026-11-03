//! Extracts one verified archive and applies its dependency's patches,
//! mirroring how FetchContent stages sources for the native build: the
//! archive's single top-level directory is stripped, and patches are applied
//! at the tree root through the build's own `apply_patch_if_needed.cmake`, so
//! the patch markers it writes are the same ones the native build writes.

use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::digest_bytes;

use super::hash_manifest::count_regular_files;
use super::patch_pipeline::{PatchStep, resolve_patch_path};
use super::pinned_archives::PinnedArchive;
use super::step_log::log_line;

const APPLY_PATCH_HELPER_PATH: &str =
    "crates/runtime-integration/native/apply_patch_if_needed.cmake";

pub(crate) struct ExtractionRequest<'request> {
    pub(crate) repository_root: &'request Path,
    pub(crate) cache_directory: &'request Path,
    pub(crate) staging_directory: &'request Path,
    pub(crate) pinned_archive: &'request PinnedArchive,
    pub(crate) dependency_description: &'static str,
    pub(crate) extraction_tree_name: &'static str,
    pub(crate) dependency_name: &'static str,
}

pub(crate) fn extract_and_patch_archive(
    extraction_request: &ExtractionRequest<'_>,
    patch_steps: &[PatchStep],
) -> Result<usize, String> {
    let archive_path = extraction_request
        .cache_directory
        .join(&extraction_request.pinned_archive.archive_file_name);
    require_verified_archive(&archive_path, extraction_request)?;
    let archive_staging_directory = extraction_request.staging_directory.join(format!(
        "{}.archive",
        extraction_request.extraction_tree_name
    ));
    let tree_directory = extraction_request
        .staging_directory
        .join(extraction_request.extraction_tree_name);
    create_directory(&archive_staging_directory)?;
    run_edge(
        Command::new("tar")
            .arg("-xzf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&archive_staging_directory),
        &format!(
            "failed to extract the {} archive",
            extraction_request.dependency_description
        ),
    )?;
    let top_level_directory =
        single_top_level_directory(&archive_staging_directory, extraction_request)?;
    if let Err(move_error) = std::fs::rename(&top_level_directory, &tree_directory) {
        return Err(format!(
            "could not stage the {} tree: {move_error}",
            extraction_request.dependency_description
        ));
    }
    let applied_patch_count =
        apply_dependency_patches(extraction_request, &tree_directory, patch_steps)?;
    let extracted_file_count = count_regular_files(&tree_directory)?;
    log_line(&format!(
        "dependency={} patches={applied_patch_count} files={extracted_file_count}",
        extraction_request.dependency_name
    ));
    Ok(extracted_file_count)
}

/// A missing, symlinked, or altered archive is refused before extraction;
/// this tool never downloads, so the fix is always the bootstrap.
fn require_verified_archive(
    archive_path: &Path,
    extraction_request: &ExtractionRequest<'_>,
) -> Result<(), String> {
    let is_regular_file = match std::fs::symlink_metadata(archive_path) {
        Ok(archive_metadata) => archive_metadata.file_type().is_file(),
        Err(_) => false,
    };
    let actual_sha256 = if is_regular_file {
        match std::fs::read(archive_path) {
            Ok(archive_bytes) => digest_bytes(&archive_bytes),
            Err(_) => String::new(),
        }
    } else {
        String::new()
    };
    if actual_sha256 == extraction_request.pinned_archive.archive_sha256 {
        return Ok(());
    }
    Err(format!(
        "{} archive is missing or fails its pinned SHA-256: {}\nrun cargo xtask bootstrap-native-dependencies to provision the verified cache",
        extraction_request.dependency_description,
        archive_path.display()
    ))
}

fn single_top_level_directory(
    archive_staging_directory: &Path,
    extraction_request: &ExtractionRequest<'_>,
) -> Result<PathBuf, String> {
    let mut top_level_entries = Vec::new();
    if let Ok(directory_entries) = std::fs::read_dir(archive_staging_directory) {
        for directory_entry in directory_entries.flatten() {
            top_level_entries.push(directory_entry.path());
        }
    }
    match top_level_entries.as_slice() {
        [top_level_entry] if top_level_entry.is_dir() => Ok(top_level_entry.clone()),
        _ => Err(format!(
            "{} archive must contain exactly one top-level directory",
            extraction_request.dependency_description
        )),
    }
}

fn apply_dependency_patches(
    extraction_request: &ExtractionRequest<'_>,
    tree_directory: &Path,
    patch_steps: &[PatchStep],
) -> Result<usize, String> {
    let patch_executable = patch_executable_path()?;
    let mut applied_patch_count = 0;
    for patch_step in patch_steps {
        if patch_step.dependency_name != extraction_request.dependency_name {
            continue;
        }
        let patch_file =
            resolve_patch_path(extraction_request.repository_root, &patch_step.patch_path);
        run_edge(
            Command::new("cmake")
                .current_dir(tree_directory)
                .arg(format!("-DPATCH_EXECUTABLE={}", patch_executable.display()))
                .arg(format!("-DPATCH_FILE={}", patch_file.display()))
                .arg("-P")
                .arg(
                    extraction_request
                        .repository_root
                        .join(APPLY_PATCH_HELPER_PATH),
                ),
            &format!(
                "patch {} does not apply to {}",
                patch_step.patch_path, extraction_request.dependency_name
            ),
        )?;
        applied_patch_count += 1;
    }
    if applied_patch_count == 0 {
        return Err(format!(
            "patch pipeline declares no patches for {}",
            extraction_request.dependency_name
        ));
    }
    Ok(applied_patch_count)
}

/// The CMake helper needs an absolute `patch`; this resolves it from PATH as
/// `command -v patch` did.
fn patch_executable_path() -> Result<PathBuf, String> {
    let search_path = std::env::var("PATH").unwrap_or_default();
    for search_directory in search_path.split(':') {
        let candidate_path = Path::new(search_directory).join("patch");
        if !search_directory.is_empty() && candidate_path.is_file() {
            return Ok(candidate_path);
        }
    }
    Err("required command is unavailable: patch".to_owned())
}

fn run_edge(edge_command: &mut Command, failure_message: &str) -> Result<(), String> {
    match edge_command.output() {
        Ok(edge_output) if edge_output.status.success() => Ok(()),
        Ok(edge_output) => Err(format!(
            "{failure_message}: {}",
            String::from_utf8_lossy(&edge_output.stderr).trim()
        )),
        Err(spawn_error) => Err(format!("{failure_message}: {spawn_error}")),
    }
}

pub(crate) fn create_directory(directory_path: &Path) -> Result<(), String> {
    match std::fs::create_dir_all(directory_path) {
        Ok(()) => Ok(()),
        Err(create_error) => Err(format!(
            "could not create {}: {create_error}",
            directory_path.display()
        )),
    }
}
