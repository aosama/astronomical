//! Portable path validation for durable provider manifests.

use std::{
    collections::BTreeSet,
    path::{Component, Path},
};

const MAXIMUM_RELATIVE_PATH_BYTES: usize = 1_024;

pub(super) fn is_safe_relative_path(relative_path: &str) -> bool {
    if relative_path.is_empty()
        || relative_path.len() > MAXIMUM_RELATIVE_PATH_BYTES
        || !relative_path.is_ascii()
        || relative_path.contains('\\')
        || relative_path.chars().any(char::is_control)
    {
        return false;
    }
    let path = Path::new(relative_path);
    if path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return false;
    }
    // Provider paths become portable on-disk identities, so aliases are rejected
    // before case-insensitive collision detection.
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
        == relative_path
}

pub(super) fn has_path_hierarchy_conflict(paths: &BTreeSet<String>, candidate_path: &str) -> bool {
    let mut ancestor_end = 0_usize;
    while let Some(relative_separator_index) = candidate_path[ancestor_end..].find('/') {
        ancestor_end = ancestor_end.saturating_add(relative_separator_index);
        if paths.contains(&candidate_path[..ancestor_end]) {
            return true;
        }
        ancestor_end = ancestor_end.saturating_add(1);
    }
    let descendant_prefix = format!("{candidate_path}/");
    paths
        .range(descendant_prefix.clone()..)
        .next()
        .is_some_and(|path| path.starts_with(&descendant_prefix))
}
