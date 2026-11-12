//! Which installed model answers a request of a helper kind (decisions or
//! speech) when the caller names a model, or names none.
//!
//! A named model must be an installed model of the kind. With no name, the
//! resident model answers when it is of the kind, so a caller who just used
//! one keeps using it without a swap. Otherwise the kind's built-in default
//! answers when it is installed. Failing that, the only installed model of
//! the kind answers, and several are ambiguous: guessing would answer with a
//! model the caller did not pick.

use astronomical_config::{DiscoveredModel, ModelCapabilities};

/// A kind of helper model: how to recognise one, and its built-in default.
pub(crate) struct HelperModelKind {
    pub(crate) is_of_kind: fn(&ModelCapabilities) -> bool,
    pub(crate) builtin_default_model_id: &'static str,
}

/// Why no model of the kind can be chosen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum HelperModelResolutionError {
    ModelNotFound,
    ModelOfAnotherKind,
    NoneInstalled,
    SeveralInstalled { installed_model_ids: Vec<String> },
}

pub(crate) fn resolve_helper_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
    requested_model: Option<&str>,
    helper_model_kind: &HelperModelKind,
) -> Result<String, HelperModelResolutionError> {
    match requested_model {
        Some(requested_model) => {
            resolve_named_helper_model(discovered_models, requested_model, helper_model_kind)
        }
        None => {
            resolve_unnamed_helper_model(discovered_models, resident_model_id, helper_model_kind)
        }
    }
}

fn resolve_named_helper_model(
    discovered_models: &[DiscoveredModel],
    requested_model: &str,
    helper_model_kind: &HelperModelKind,
) -> Result<String, HelperModelResolutionError> {
    let known_model_ids: Vec<&str> = discovered_models
        .iter()
        .map(|discovered_model| discovered_model.model_id.as_str())
        .collect();
    let resolved_model_id =
        astronomical_config::resolve_model_id(requested_model, &known_model_ids);
    let Some(discovered_model) = discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == resolved_model_id)
    else {
        return Err(HelperModelResolutionError::ModelNotFound);
    };
    if (helper_model_kind.is_of_kind)(&discovered_model.capabilities) {
        return Ok(discovered_model.model_id.clone());
    }
    Err(HelperModelResolutionError::ModelOfAnotherKind)
}

fn resolve_unnamed_helper_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
    helper_model_kind: &HelperModelKind,
) -> Result<String, HelperModelResolutionError> {
    let mut installed_model_ids: Vec<String> = discovered_models
        .iter()
        .filter(|discovered_model| (helper_model_kind.is_of_kind)(&discovered_model.capabilities))
        .map(|discovered_model| discovered_model.model_id.clone())
        .collect();
    let preferred_model_ids = [
        resident_model_id,
        Some(helper_model_kind.builtin_default_model_id),
    ];
    for preferred_model_id in preferred_model_ids.into_iter().flatten() {
        if installed_model_ids
            .iter()
            .any(|installed_model_id| installed_model_id == preferred_model_id)
        {
            return Ok(preferred_model_id.to_owned());
        }
    }
    match installed_model_ids.len() {
        0 => Err(HelperModelResolutionError::NoneInstalled),
        1 => Ok(installed_model_ids.remove(0)),
        _ => {
            installed_model_ids.sort();
            Err(HelperModelResolutionError::SeveralInstalled {
                installed_model_ids,
            })
        }
    }
}
