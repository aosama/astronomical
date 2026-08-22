use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::{DiscoveredModelError, ModelFamily, classify_model_directory};
use crate::{decode_huggingface_cache_directory_name, leaf_model_id};

const MAXIMUM_CLASSIFIED_SCAN_DEPTH: usize = 4;
const MAXIMUM_REVISION_METADATA_BYTES: u64 = 4_096;
const MAXIMUM_LIBRARY_PROVENANCE_BYTES: u64 = 1_024;
const LIBRARY_PROVENANCE_FILE_NAME: &str = ".astronomical-library-provenance.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryModelProvenance {
    schema_version: u32,
    provider_model_id: String,
    revision: String,
}

/// One classified artifact candidate, including provenance unavailable to executable discovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassifiedModelArtifact {
    pub model_id: String,
    pub upstream_revision: Option<String>,
    pub model_directory: PathBuf,
    pub model_family: ModelFamily,
}

/// Finds classified model-family artifacts without advertising them as executable models.
pub fn discover_classified_model_artifacts(
    model_directories: &[PathBuf],
) -> Result<Vec<ClassifiedModelArtifact>, DiscoveredModelError> {
    let mut classified_artifacts = Vec::new();
    for model_directory in model_directories {
        scan_classified_artifacts(model_directory, 0, true, &mut classified_artifacts)?;
    }
    classified_artifacts.sort_by(|first_artifact, second_artifact| {
        first_artifact
            .model_id
            .cmp(&second_artifact.model_id)
            .then_with(|| {
                first_artifact
                    .upstream_revision
                    .cmp(&second_artifact.upstream_revision)
            })
            .then_with(|| {
                first_artifact
                    .model_directory
                    .cmp(&second_artifact.model_directory)
            })
    });
    Ok(classified_artifacts)
}

fn scan_classified_artifacts(
    current_directory: &Path,
    depth: usize,
    is_configured_root: bool,
    classified_artifacts: &mut Vec<ClassifiedModelArtifact>,
) -> Result<(), DiscoveredModelError> {
    if depth > MAXIMUM_CLASSIFIED_SCAN_DEPTH {
        return Ok(());
    }
    if current_directory.join("config.json").is_file()
        || current_directory.join("model_index.json").is_file()
    {
        if let Ok(Some(model_family)) = classify_model_directory(current_directory)
            && let Some(model_id) = model_identity(current_directory)
        {
            classified_artifacts.push(ClassifiedModelArtifact {
                model_id,
                upstream_revision: immutable_model_revision(current_directory),
                model_directory: current_directory.to_path_buf(),
                model_family,
            });
        }
        return Ok(());
    }
    let directory_entries = match fs::read_dir(current_directory) {
        Ok(directory_entries) => directory_entries,
        Err(source) if is_configured_root => {
            return Err(DiscoveredModelError::ReadDirectory {
                directory_path: current_directory.to_path_buf(),
                source,
            });
        }
        Err(_) => return Ok(()),
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        if entry_path.is_dir()
            && !entry_path
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .is_some_and(|file_name| file_name.starts_with('.'))
        {
            scan_classified_artifacts(&entry_path, depth + 1, false, classified_artifacts)?;
        }
    }
    Ok(())
}

/// Serving identity for a classified artifact: Hugging Face leaf ID or directory name.
#[must_use]
pub fn requestable_model_id(model_directory: &Path) -> Option<String> {
    for ancestor in model_directory.ancestors() {
        let Some(directory_name) = ancestor
            .file_name()
            .and_then(|file_name| file_name.to_str())
        else {
            continue;
        };
        if let Some(decoded_model_id) = decode_huggingface_cache_directory_name(directory_name) {
            return Some(leaf_model_id(&decoded_model_id).to_owned());
        }
    }
    model_directory
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .map(str::to_owned)
}

fn model_identity(model_directory: &Path) -> Option<String> {
    for ancestor in model_directory.ancestors() {
        let Some(directory_name) = ancestor
            .file_name()
            .and_then(|file_name| file_name.to_str())
        else {
            continue;
        };
        if let Some(model_id) = decode_huggingface_cache_directory_name(directory_name) {
            return Some(model_id);
        }
    }
    let repository_name = model_directory.file_name()?.to_str()?;
    let organization_name = model_directory.parent()?.file_name()?.to_str()?;
    Some(format!("{organization_name}/{repository_name}"))
}

pub(super) fn immutable_model_provenance(model_directory: &Path) -> Option<(String, String)> {
    let provenance_file_path = model_directory.join(LIBRARY_PROVENANCE_FILE_NAME);
    let provenance_metadata = fs::symlink_metadata(&provenance_file_path).ok()?;
    if !provenance_metadata.is_file()
        || provenance_metadata.len() > MAXIMUM_LIBRARY_PROVENANCE_BYTES
    {
        return None;
    }
    let provenance_bytes = fs::read(provenance_file_path).ok()?;
    let provenance: LibraryModelProvenance = serde_json::from_slice(&provenance_bytes).ok()?;
    let has_valid_revision = provenance.revision.len() == 40
        && provenance.revision.bytes().all(|revision_byte| {
            revision_byte.is_ascii_digit() || (b'a'..=b'f').contains(&revision_byte)
        });
    if provenance.schema_version != 1
        || !has_valid_revision
        || !is_valid_provider_model_id(&provenance.provider_model_id)
        || immutable_model_revision(model_directory).as_deref() != Some(&provenance.revision)
    {
        return None;
    }
    Some((provenance.provider_model_id, provenance.revision))
}

fn is_valid_provider_model_id(provider_model_id: &str) -> bool {
    let mut identity_components = provider_model_id.split('/');
    let Some(organization) = identity_components.next() else {
        return false;
    };
    let Some(repository) = identity_components.next() else {
        return false;
    };
    identity_components.next().is_none()
        && [organization, repository].into_iter().all(|component| {
            !component.is_empty()
                && component != "."
                && component != ".."
                && component.bytes().all(|component_byte| {
                    component_byte.is_ascii_alphanumeric()
                        || matches!(component_byte, b'-' | b'_' | b'.')
                })
        })
}

/// Returns artifact provenance only when its source records an immutable revision candidate.
pub(super) fn immutable_model_revision(model_directory: &Path) -> Option<String> {
    let authoritative_file_name = if model_directory.join("model_index.json").is_file() {
        "model_index.json"
    } else {
        "config.json"
    };
    immutable_file_revision(model_directory, authoritative_file_name)
}

pub(super) fn immutable_file_revision(
    model_directory: &Path,
    authoritative_file_name: &str,
) -> Option<String> {
    let local_metadata_path = model_directory.join(format!(
        ".cache/huggingface/download/{authoritative_file_name}.metadata"
    ));
    if local_metadata_path
        .metadata()
        .ok()
        .is_some_and(|metadata| metadata.len() <= MAXIMUM_REVISION_METADATA_BYTES)
    {
        return fs::read_to_string(local_metadata_path)
            .ok()?
            .lines()
            .next()
            .map(str::to_owned);
    }
    model_directory
        .ancestors()
        .any(|ancestor| {
            ancestor
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .and_then(decode_huggingface_cache_directory_name)
                .is_some()
        })
        .then(|| model_directory.file_name()?.to_str().map(str::to_owned))
        .flatten()
}
