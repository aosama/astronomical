//! What a person types to name a model: a local directory, or a Hugging Face
//! repository with an optional revision.

use std::path::{Path, PathBuf};

use crate::error::VisionTowerError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactReference {
    LocalDirectory(PathBuf),
    HuggingFaceRepository {
        repository_id: String,
        revision: Option<String>,
    },
}

impl ArtifactReference {
    /// An existing directory wins over the repository reading, because a person
    /// who has a folder with that name in front of them means the folder.
    pub fn parse(reference_text: &str) -> Result<Self, VisionTowerError> {
        if Path::new(reference_text).is_dir() {
            return Ok(Self::LocalDirectory(PathBuf::from(reference_text)));
        }
        let (repository_id, revision) = match reference_text.split_once('@') {
            Some((repository_id, revision)) => (repository_id, Some(revision)),
            None => (reference_text, None),
        };
        let is_revision_valid = match revision {
            Some(revision) => is_valid_name_segment(revision),
            None => true,
        };
        if !is_valid_repository_id(repository_id) || !is_revision_valid {
            return Err(VisionTowerError::UnrecognizedModelReference {
                reference: reference_text.to_owned(),
            });
        }
        Ok(Self::HuggingFaceRepository {
            repository_id: repository_id.to_owned(),
            revision: revision.map(str::to_owned),
        })
    }
}

fn is_valid_repository_id(repository_id: &str) -> bool {
    match repository_id.split_once('/') {
        Some((organization, model_name)) => {
            is_valid_name_segment(organization) && is_valid_name_segment(model_name)
        }
        None => false,
    }
}

/// Refusing anything else keeps a typed reference from reshaping the request URL.
fn is_valid_name_segment(segment: &str) -> bool {
    if segment.is_empty() || segment.starts_with('.') {
        return false;
    }
    for character in segment.chars() {
        let is_allowed = character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.');
        if !is_allowed {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "artifact_reference_tests.rs"]
mod tests;
