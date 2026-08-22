//! Validated release-authored selection of executable files from an immutable repository tree.

use super::hugging_face_hub_bounds::is_canonical_ascii_path;

const MAXIMUM_INCLUDED_PATH_COUNT: usize = 64;
const MAXIMUM_INCLUDED_PATH_BYTES: usize = 1_024;

/// Exact files and directory prefixes that belong to one executable model package.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DownloadPathSelection {
    included_paths: Option<Vec<String>>,
}

impl DownloadPathSelection {
    pub(crate) fn try_new(included_paths: Option<Vec<String>>) -> Result<Self, ()> {
        let Some(included_paths) = included_paths else {
            return Ok(Self::default());
        };
        if included_paths.is_empty() || included_paths.len() > MAXIMUM_INCLUDED_PATH_COUNT {
            return Err(());
        }
        let mut validated_paths: Vec<String> = Vec::with_capacity(included_paths.len());
        for included_path in included_paths {
            if !is_valid_included_path(&included_path) {
                return Err(());
            }
            let normalized_path = included_path.to_ascii_lowercase();
            if validated_paths
                .iter()
                .any(|existing_path| selectors_overlap(existing_path, &normalized_path))
            {
                return Err(());
            }
            validated_paths.push(normalized_path);
        }
        Ok(Self {
            included_paths: Some(validated_paths),
        })
    }

    /// Returns whether a validated Hub file belongs to the executable package.
    #[must_use]
    pub fn includes(&self, relative_path: &str) -> bool {
        self.included_paths.as_ref().is_none_or(|included_paths| {
            let normalized_path = relative_path.to_ascii_lowercase();
            included_paths.iter().any(|included_path| {
                if included_path.ends_with('/') {
                    normalized_path.starts_with(included_path)
                } else {
                    normalized_path == *included_path
                }
            })
        })
    }
}

fn is_valid_included_path(included_path: &str) -> bool {
    if included_path.is_empty()
        || included_path.len() > MAXIMUM_INCLUDED_PATH_BYTES
        || !included_path.is_ascii()
        || included_path.contains('\\')
        || included_path.chars().any(char::is_control)
        || included_path.starts_with('/')
    {
        return false;
    }
    let path_without_directory_marker = included_path.strip_suffix('/').unwrap_or(included_path);
    if path_without_directory_marker.is_empty() || path_without_directory_marker.ends_with('/') {
        return false;
    }
    is_canonical_ascii_path(path_without_directory_marker, MAXIMUM_INCLUDED_PATH_BYTES)
}

fn selectors_overlap(first_path: &str, second_path: &str) -> bool {
    let first_base = first_path.trim_end_matches('/');
    let second_base = second_path.trim_end_matches('/');
    first_base == second_base
        || first_path.ends_with('/') && second_path.starts_with(first_path)
        || second_path.ends_with('/') && first_path.starts_with(second_path)
}
