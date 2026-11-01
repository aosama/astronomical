use std::path::Path;

// These spellings mirror the constants in the astronomical-quantization
// crate's artifact module. That crate is an optional dependency of this one
// (enabled only with direct-mlx), and artifact validation must locate the
// metadata in every build, so the values are restated here rather than
// imported.
pub(crate) const OPTIQ_METADATA_RELATIVE_PATH: &str = "optiq/metadata.json";
/// The pre-0.4.18 root-level name that published quants still carry.
pub(crate) const LEGACY_OPTIQ_METADATA_RELATIVE_PATH: &str = "optiq_metadata.json";

/// Picks the metadata location from which candidates exist. The canonical
/// folder wins because a current writer that also leaves a stale legacy copy
/// behind updated the canonical one.
pub(crate) fn choose_optiq_metadata_relative_path(
    has_canonical_metadata: bool,
    has_legacy_metadata: bool,
) -> Option<&'static str> {
    if has_canonical_metadata {
        Some(OPTIQ_METADATA_RELATIVE_PATH)
    } else if has_legacy_metadata {
        Some(LEGACY_OPTIQ_METADATA_RELATIVE_PATH)
    } else {
        None
    }
}

/// Returns the model-directory-relative path of the OptiQ metadata file that
/// exists on disk, or `None` for artifacts that carry none.
pub fn resolve_optiq_metadata_relative_path(model_directory: &Path) -> Option<&'static str> {
    choose_optiq_metadata_relative_path(
        model_directory.join(OPTIQ_METADATA_RELATIVE_PATH).is_file(),
        model_directory
            .join(LEGACY_OPTIQ_METADATA_RELATIVE_PATH)
            .is_file(),
    )
}

#[cfg(test)]
mod metadata_location_tests;
