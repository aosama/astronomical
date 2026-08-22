//! Pagination, portable path, and durable-metadata bounds for Hub manifest discovery.

use std::collections::BTreeSet;

use super::{HubManifestFile, HuggingFaceHubError};

const MAXIMUM_PAGINATION_URL_BYTES: usize = 8_192;
const DOWNLOAD_JOB_FILE_FIXED_METADATA_ALLOWANCE_BYTES: usize = 256;

pub(super) fn parse_next_link(
    link_header: Option<&str>,
) -> Result<Option<String>, HuggingFaceHubError> {
    let Some(link_header) = link_header else {
        return Ok(None);
    };
    let mut next_url = None;
    for link_segment in link_header.split(',') {
        let mut segment_parts = link_segment.trim().split(';');
        let target = segment_parts
            .next()
            .ok_or(HuggingFaceHubError::UnsafePaginationLink)?
            .trim();
        let is_next = segment_parts.any(|parameter| {
            let parameter = parameter.trim();
            parameter == "rel=next" || parameter == "rel=\"next\""
        });
        if is_next {
            if next_url.is_some() || !target.starts_with('<') || !target.ends_with('>') {
                return Err(HuggingFaceHubError::UnsafePaginationLink);
            }
            next_url = Some(target[1..target.len() - 1].to_owned());
        }
    }
    Ok(next_url)
}

pub(super) fn validate_tree_page_url(
    page_url: &str,
    expected_tree_url: &str,
) -> Result<(), HuggingFaceHubError> {
    let Some(query) = page_url.strip_prefix(expected_tree_url) else {
        return Err(HuggingFaceHubError::UnsafePaginationLink);
    };
    if page_url.len() > MAXIMUM_PAGINATION_URL_BYTES
        || !page_url.is_ascii()
        || page_url.contains('#')
        || !query.starts_with('?')
        || !query[1..]
            .split('&')
            .any(|parameter| parameter == "recursive=true")
    {
        return Err(HuggingFaceHubError::UnsafePaginationLink);
    }
    Ok(())
}

pub(super) fn is_canonical_ascii_path(relative_path: &str, maximum_path_bytes: usize) -> bool {
    !relative_path.is_empty()
        && relative_path.len() <= maximum_path_bytes
        && relative_path.is_ascii()
        && !relative_path.contains('\\')
        && !relative_path
            .bytes()
            .any(|character| character.is_ascii_control())
        && relative_path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

pub(super) fn has_ancestor_path(file_paths: &BTreeSet<String>, candidate_path: &str) -> bool {
    candidate_path
        .match_indices('/')
        .any(|(separator_index, _)| file_paths.contains(&candidate_path[..separator_index]))
}

pub(super) fn has_descendant_path(all_paths: &BTreeSet<String>, candidate_path: &str) -> bool {
    let descendant_prefix = format!("{candidate_path}/");
    all_paths
        .range(descendant_prefix.clone()..)
        .next()
        .is_some_and(|path| path.starts_with(&descendant_prefix))
}

pub(super) fn estimated_durable_file_metadata_bytes(manifest_file: &HubManifestFile) -> usize {
    manifest_file
        .relative_path()
        .len()
        .saturating_mul(2)
        .saturating_add(manifest_file.digest().hex().len())
        .saturating_add(DOWNLOAD_JOB_FILE_FIXED_METADATA_ALLOWANCE_BYTES)
}
