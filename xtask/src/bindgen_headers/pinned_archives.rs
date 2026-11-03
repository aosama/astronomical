//! Picks the MLX and MLX-C archives out of the pinned native dependency
//! manifest. metal-cpp, nlohmann/json, and fmt are build-time dependencies of
//! the MLX C++ build and contribute no bindgen-relevant headers.

use crate::native_dependency_manifest::parse_manifest_line;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedArchive {
    pub(crate) archive_file_name: String,
    pub(crate) archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedHeaderArchives {
    pub(crate) mlx: PinnedArchive,
    pub(crate) mlx_c: PinnedArchive,
}

/// Validates every line with the bootstrap's manifest policy, then selects
/// by description. "MLX C " is tested before "MLX " because "MLX" is a strict
/// prefix of "MLX C". The archive count is not enforced here, so fixture
/// manifests in tests can carry just the two archives that matter.
pub(crate) fn select_pinned_header_archives(
    manifest_text: &str,
) -> Result<PinnedHeaderArchives, String> {
    let mut mlx_archive = None;
    let mut mlx_c_archive = None;
    for manifest_line in manifest_text.lines() {
        if manifest_line.trim().is_empty() {
            continue;
        }
        let manifest_entry = parse_manifest_line(manifest_line)?;
        let pinned_archive = PinnedArchive {
            archive_file_name: manifest_entry.archive_file_name,
            archive_sha256: manifest_entry.archive_sha256,
        };
        if manifest_entry.dependency_description.starts_with("MLX C ") {
            if mlx_c_archive.is_some() {
                return Err("pinned archive manifest declares the MLX C archive twice".to_owned());
            }
            mlx_c_archive = Some(pinned_archive);
        } else if manifest_entry.dependency_description.starts_with("MLX ") {
            if mlx_archive.is_some() {
                return Err("pinned archive manifest declares the MLX archive twice".to_owned());
            }
            mlx_archive = Some(pinned_archive);
        }
    }
    match (mlx_archive, mlx_c_archive) {
        (Some(mlx), Some(mlx_c)) => Ok(PinnedHeaderArchives { mlx, mlx_c }),
        _ => Err(
            "pinned archive manifest must declare both the MLX and the MLX C archives".to_owned(),
        ),
    }
}
