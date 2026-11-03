//! One provisioning pass: reuse the published extraction when its recorded
//! hash manifests still match, otherwise extract into a staging directory
//! and publish it with one rename. The completion marker is written last, so
//! a partially written extraction never verifies as complete.

use std::path::{Path, PathBuf};

use super::archive_extraction::{ExtractionRequest, create_directory, extract_and_patch_archive};
use super::hash_manifest::tree_hash_manifest;
use super::header_locations::{
    COMPLETION_MARKER_FILE_NAME, EXTRACTION_TREE_MLX, EXTRACTION_TREE_MLX_C,
    HEADERS_CACHE_SUBDIRECTORY, extraction_directory, source_identity,
};
use super::options::ProvisionOptions;
use super::patch_pipeline::PatchStep;
use super::pinned_archives::{PinnedHeaderArchives, select_pinned_header_archives};
use crate::bootstrap_native_dependencies::read_pinned_manifest_text;
use crate::step_log::{log_line, start_step};

pub(crate) fn provision_headers(
    repository_root: &Path,
    provision_options: &ProvisionOptions,
    cache_directory: &Path,
    patch_steps: &[PatchStep],
) -> Result<(), String> {
    let headers_root_directory = cache_directory.join(HEADERS_CACHE_SUBDIRECTORY);
    create_private_directory(&headers_root_directory)?;
    let read_manifest_step = start_step("read-pinned-manifest");
    let manifest_text = pinned_manifest_text(repository_root, provision_options)?;
    let pinned_archives = select_pinned_header_archives(&manifest_text)?;
    read_manifest_step.finish("success");
    let identity_step = start_step("resolve-source-identity");
    let source_identity = source_identity(repository_root)?;
    let extraction_directory = extraction_directory(cache_directory, &source_identity);
    identity_step.finish("success");
    log_line(&format!(
        "extraction_directory={}",
        extraction_directory.display()
    ));
    let verify_step = start_step("verify-existing-extraction");
    if extraction_is_intact(&extraction_directory)? {
        log_line(&format!(
            "extraction_directory={} status=verified-cached",
            extraction_directory.display()
        ));
        verify_step.finish("success");
        return Ok(());
    }
    if extraction_directory.exists() {
        log_line(&format!(
            "extraction_directory={} status=re-extracting reason=missing-or-modified",
            extraction_directory.display()
        ));
    }
    verify_step.finish("success");
    let publication = HeaderPublication {
        repository_root,
        cache_directory,
        source_identity,
        pinned_archives,
    };
    publication.extract_and_publish(&headers_root_directory, &extraction_directory, patch_steps)
}

fn pinned_manifest_text(
    repository_root: &Path,
    provision_options: &ProvisionOptions,
) -> Result<String, String> {
    match &provision_options.explicit_manifest_path {
        Some(explicit_manifest_path) => match std::fs::read_to_string(explicit_manifest_path) {
            Ok(manifest_text) => Ok(manifest_text),
            Err(read_error) => Err(format!(
                "could not read {explicit_manifest_path}: {read_error}"
            )),
        },
        None => read_pinned_manifest_text(repository_root),
    }
}

/// The marker alone is not trusted: both trees are re-hashed against their
/// recorded manifests, so an edited or truncated header self-heals.
fn extraction_is_intact(extraction_directory: &Path) -> Result<bool, String> {
    if !extraction_directory
        .join(COMPLETION_MARKER_FILE_NAME)
        .is_file()
    {
        return Ok(false);
    }
    for (tree_name, manifest_name) in [
        (EXTRACTION_TREE_MLX, "mlx.manifest"),
        (EXTRACTION_TREE_MLX_C, "mlx_c.manifest"),
    ] {
        let Ok(recorded_manifest) =
            std::fs::read_to_string(extraction_directory.join(manifest_name))
        else {
            return Ok(false);
        };
        let tree_directory = extraction_directory.join(tree_name);
        if !tree_directory.is_dir() || tree_hash_manifest(&tree_directory)? != recorded_manifest {
            return Ok(false);
        }
    }
    Ok(true)
}

struct HeaderPublication<'publication> {
    repository_root: &'publication Path,
    cache_directory: &'publication Path,
    source_identity: String,
    pinned_archives: PinnedHeaderArchives,
}

impl HeaderPublication<'_> {
    fn extract_and_publish(
        &self,
        headers_root_directory: &Path,
        extraction_directory: &Path,
        patch_steps: &[PatchStep],
    ) -> Result<(), String> {
        let staging_directory = unique_staging_directory(headers_root_directory);
        create_directory(&staging_directory)?;
        let publication_result =
            self.stage_and_publish(&staging_directory, extraction_directory, patch_steps);
        if publication_result.is_err() {
            // A failed pass must not leave a half-built staging tree behind.
            let _ = std::fs::remove_dir_all(&staging_directory);
        }
        publication_result
    }

    fn stage_and_publish(
        &self,
        staging_directory: &Path,
        extraction_directory: &Path,
        patch_steps: &[PatchStep],
    ) -> Result<(), String> {
        let extract_step = start_step("extract-and-patch-archives");
        let mlx_file_count = extract_and_patch_archive(
            &self.extraction_request(staging_directory, "MLX", EXTRACTION_TREE_MLX, "mlx"),
            patch_steps,
        )?;
        let mlx_c_file_count = extract_and_patch_archive(
            &self.extraction_request(staging_directory, "MLX C", EXTRACTION_TREE_MLX_C, "mlx_c"),
            patch_steps,
        )?;
        extract_step.finish("success");
        let publish_step = start_step("publish-extraction");
        write_text(
            &staging_directory.join("mlx.manifest"),
            &tree_hash_manifest(&staging_directory.join(EXTRACTION_TREE_MLX))?,
        )?;
        write_text(
            &staging_directory.join("mlx_c.manifest"),
            &tree_hash_manifest(&staging_directory.join(EXTRACTION_TREE_MLX_C))?,
        )?;
        write_text(
            &staging_directory.join(COMPLETION_MARKER_FILE_NAME),
            &completion_marker_text(
                &self.source_identity,
                &self.pinned_archives,
                mlx_file_count,
                mlx_c_file_count,
            ),
        )?;
        replace_directory(staging_directory, extraction_directory)?;
        publish_step.finish("success");
        log_line(&format!(
            "extraction_directory={} status=provisioned mlx_files={mlx_file_count} mlx_c_files={mlx_c_file_count}",
            extraction_directory.display()
        ));
        Ok(())
    }

    fn extraction_request<'request>(
        &'request self,
        staging_directory: &'request Path,
        dependency_description: &'static str,
        extraction_tree_name: &'static str,
        dependency_name: &'static str,
    ) -> ExtractionRequest<'request> {
        let pinned_archive = if dependency_name == "mlx" {
            &self.pinned_archives.mlx
        } else {
            &self.pinned_archives.mlx_c
        };
        ExtractionRequest {
            repository_root: self.repository_root,
            cache_directory: self.cache_directory,
            staging_directory,
            pinned_archive,
            dependency_description,
            extraction_tree_name,
            dependency_name,
        }
    }
}

pub(crate) fn completion_marker_text(
    source_identity: &str,
    pinned_archives: &PinnedHeaderArchives,
    mlx_file_count: usize,
    mlx_c_file_count: usize,
) -> String {
    format!(
        "source_identity={source_identity}\nmlx_archive={}\nmlx_archive_sha256={}\nmlx_c_archive={}\nmlx_c_archive_sha256={}\nmlx_file_count={mlx_file_count}\nmlx_c_file_count={mlx_c_file_count}\n",
        pinned_archives.mlx.archive_file_name,
        pinned_archives.mlx.archive_sha256,
        pinned_archives.mlx_c.archive_file_name,
        pinned_archives.mlx_c.archive_sha256,
    )
}

/// Pid plus nanoseconds, so a staging directory a crashed run left behind
/// never collides with the next run's.
fn unique_staging_directory(headers_root_directory: &Path) -> PathBuf {
    let nanoseconds = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(since_epoch) => since_epoch.subsec_nanos(),
        Err(_) => 0,
    };
    headers_root_directory.join(format!(".staging.{}.{nanoseconds}", std::process::id()))
}

/// The headers cache holds only this user's extraction; 0700 keeps other
/// local accounts from swapping headers under bindgen.
fn create_private_directory(directory_path: &Path) -> Result<(), String> {
    create_directory(directory_path)?;
    let private_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o700);
    match std::fs::set_permissions(directory_path, private_permissions) {
        Ok(()) => Ok(()),
        Err(permission_error) => Err(format!(
            "could not restrict {}: {permission_error}",
            directory_path.display()
        )),
    }
}

fn replace_directory(staging_directory: &Path, extraction_directory: &Path) -> Result<(), String> {
    if std::fs::symlink_metadata(extraction_directory).is_ok_and(is_symlink_metadata) {
        return Err(format!(
            "refusing to replace a symlinked extraction directory: {}",
            extraction_directory.display()
        ));
    }
    if extraction_directory.exists()
        && let Err(remove_error) = std::fs::remove_dir_all(extraction_directory)
    {
        return Err(format!(
            "could not remove the stale extraction: {remove_error}"
        ));
    }
    match std::fs::rename(staging_directory, extraction_directory) {
        Ok(()) => Ok(()),
        Err(rename_error) => Err(format!("could not publish the extraction: {rename_error}")),
    }
}

fn is_symlink_metadata(path_metadata: std::fs::Metadata) -> bool {
    path_metadata.file_type().is_symlink()
}

fn write_text(file_path: &Path, file_text: &str) -> Result<(), String> {
    match std::fs::write(file_path, file_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            file_path.display()
        )),
    }
}
