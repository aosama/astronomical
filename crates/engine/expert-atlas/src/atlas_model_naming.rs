//! The name an Atlas conversion gives its new model.
//!
//! The converted model is a model of its own, persistent in the Library
//! beside its source, so its id keeps the source's organization and marks the
//! layout in the name: `organization/model-name` becomes
//! `organization/model-name-astro-atlas-streaming`.

/// Appended to the source model's name.
pub const ATLAS_MODEL_NAME_SUFFIX: &str = "-astro-atlas-streaming";

/// The default Atlas model id for `source_model_id` (`organization/name` or a
/// bare `name`). An id that already carries the suffix keeps a single copy.
#[must_use]
pub fn atlas_model_id_for(source_model_id: &str) -> String {
    if source_model_id.ends_with(ATLAS_MODEL_NAME_SUFFIX) {
        return source_model_id.to_owned();
    }
    format!("{source_model_id}{ATLAS_MODEL_NAME_SUFFIX}")
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelIdError {
    #[error("model id {0:?} must be ORGANIZATION/NAME")]
    NotOrganizationAndName(String),
    #[error("model id {0:?} contains a path component that is not a plain name")]
    UnsafeComponent(String),
}

/// Splits `organization/name`, refusing anything that could escape the
/// Library folder (empty parts, `.`, `..`, separators inside a part).
pub fn organization_and_name(model_id: &str) -> Result<(&str, &str), ModelIdError> {
    let (organization, name) = model_id
        .split_once('/')
        .ok_or_else(|| ModelIdError::NotOrganizationAndName(model_id.to_owned()))?;
    for component in [organization, name] {
        let is_plain_name = !component.is_empty()
            && component != "."
            && component != ".."
            && !component.contains(['/', '\\', '\0']);
        if !is_plain_name {
            return Err(ModelIdError::UnsafeComponent(model_id.to_owned()));
        }
    }
    Ok((organization, name))
}

#[cfg(test)]
#[path = "atlas_model_naming_tests.rs"]
mod atlas_model_naming_tests;
