//! Catalog entry identities: what makes a Hugging Face repository name and a
//! revision acceptable, and the rule that no two entries share a name.

use std::collections::BTreeSet;

use crate::download_catalog::DownloadCatalogError;

const MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH: usize = 96;
const GIT_COMMIT_SHA_HEX_CHARACTER_COUNT: usize = 40;

/// Claims the names an entry's repository had before a rename on Hugging
/// Face. They are claimed like current names, so no other entry can take one:
/// a copy downloaded under a former name must lead to exactly one entry.
pub(crate) fn claim_former_identities(
    former_huggingface_ids: &[String],
    entry_index: usize,
    seen_identities: &mut SeenEntryIdentities,
) -> Result<(), DownloadCatalogError> {
    for former_huggingface_id in former_huggingface_ids {
        if !is_valid_huggingface_id(former_huggingface_id) {
            return Err(DownloadCatalogError::InvalidFormerHuggingFaceId { entry_index });
        }
        seen_identities.claim(former_huggingface_id, entry_index)?;
    }
    Ok(())
}

/// The identities every entry read so far has claimed, compared ignoring case.
#[derive(Default)]
pub(crate) struct SeenEntryIdentities {
    pub(crate) huggingface_ids: BTreeSet<String>,
    pub(crate) model_ids: BTreeSet<String>,
}

impl SeenEntryIdentities {
    /// Claims one entry's Hugging Face identity and model ID. A repeated
    /// identity is reported first, because it also repeats the model ID.
    pub(crate) fn claim(
        &mut self,
        huggingface_id: &str,
        entry_index: usize,
    ) -> Result<(), DownloadCatalogError> {
        if !self
            .huggingface_ids
            .insert(huggingface_id.to_ascii_lowercase())
        {
            return Err(DownloadCatalogError::DuplicateHuggingFaceId);
        }
        let model_id = config::leaf_model_id(huggingface_id);
        if !self.model_ids.insert(model_id.to_ascii_lowercase()) {
            return Err(DownloadCatalogError::DuplicateModelId { entry_index });
        }
        Ok(())
    }
}

pub fn is_valid_huggingface_id(huggingface_id: &str) -> bool {
    let mut identity_components = huggingface_id.split('/');
    let organization = identity_components.next().unwrap_or_default();
    let model_name = identity_components.next().unwrap_or_default();
    identity_components.next().is_none()
        && is_valid_hugging_face_component(organization)
        && is_valid_hugging_face_component(model_name)
        && !model_name.ends_with(".git")
}

pub fn is_valid_immutable_revision(revision: &str) -> bool {
    lowercase_hex::is_lowercase_hex_of_length(revision, GIT_COMMIT_SHA_HEX_CHARACTER_COUNT)
}

fn is_valid_hugging_face_component(component: &str) -> bool {
    let component_bytes = component.as_bytes();
    if component.is_empty()
        || component.len() > MAXIMUM_HUGGING_FACE_COMPONENT_LENGTH
        || component_bytes
            .first()
            .is_some_and(|character| matches!(*character, b'-' | b'.'))
        || component_bytes
            .last()
            .is_some_and(|character| matches!(*character, b'-' | b'.'))
        || component.contains("--")
        || component.contains("..")
    {
        return false;
    }
    component_bytes.iter().copied().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, b'-' | b'_' | b'.')
    })
}

#[cfg(test)]
#[path = "entry_identity_tests.rs"]
mod entry_identity_tests;
