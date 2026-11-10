//! Decisions logic shared by the REST endpoint and the daemon IPC service:
//! which model answers a request, and the request the worker receives.
//!
//! Both surfaces accept the same System One request and must agree on which
//! model it reaches, so that choice is made here once and each surface only
//! translates the outcome into its own response.

use astronomical_config::{DiscoveredModel, ModelCapabilities};
use astronomical_decision_contract::{
    OrderedJsonValue, QuestionDefinitionError, parse_system_one_questions,
};
use astronomical_ipc_protocol::{DecisionsCommand, RequestId};
use thiserror::Error;

/// Why no model can be chosen for a decisions request.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum DecisionsModelResolutionError {
    #[error("model is not available to the local worker")]
    ModelNotFound,
    #[error("the requested model does not answer decision requests")]
    ModelCannotDecide,
    #[error("no decision model is installed; download one from the Models page")]
    NoDecisionModelInstalled,
    #[error(
        "more than one decision model is installed ({}); name one with `model`",
        installed_model_ids.join(", ")
    )]
    SeveralDecisionModelsInstalled { installed_model_ids: Vec<String> },
}

/// Chooses the model that answers a decisions request.
///
/// A named model must be an installed decision model. With no name, the
/// resident model answers when it is a decision model, so a caller who just
/// used one keeps using it without a swap. Otherwise the built-in default
/// decision model answers when it is installed: its download installs a
/// second model beside it, so without this rule every such install would be
/// ambiguous from the first request. Failing that, the only installed
/// decision model answers, and several installed ones are ambiguous:
/// guessing would answer with a model the caller did not pick.
pub(crate) fn resolve_decisions_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
    requested_model: Option<&str>,
) -> Result<String, DecisionsModelResolutionError> {
    match requested_model {
        Some(requested_model) => resolve_named_decisions_model(discovered_models, requested_model),
        None => resolve_unnamed_decisions_model(discovered_models, resident_model_id),
    }
}

/// A named model answers only when it is installed and is a decision model.
fn resolve_named_decisions_model(
    discovered_models: &[DiscoveredModel],
    requested_model: &str,
) -> Result<String, DecisionsModelResolutionError> {
    let mut known_model_ids = Vec::with_capacity(discovered_models.len());
    for discovered_model in discovered_models {
        known_model_ids.push(discovered_model.model_id.as_str());
    }
    let resolved_model_id =
        astronomical_config::resolve_model_id(requested_model, &known_model_ids);
    for discovered_model in discovered_models {
        if discovered_model.model_id == resolved_model_id {
            return match discovered_model.capabilities {
                ModelCapabilities::Decisions(_) => Ok(discovered_model.model_id.clone()),
                _ => Err(DecisionsModelResolutionError::ModelCannotDecide),
            };
        }
    }
    Err(DecisionsModelResolutionError::ModelNotFound)
}

/// With no model named: the resident decision model, then the built-in
/// default, then the only installed one.
fn resolve_unnamed_decisions_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
) -> Result<String, DecisionsModelResolutionError> {
    let mut installed_decision_model_ids = Vec::new();
    for discovered_model in discovered_models {
        if matches!(
            discovered_model.capabilities,
            ModelCapabilities::Decisions(_)
        ) {
            installed_decision_model_ids.push(discovered_model.model_id.clone());
        }
    }
    let preferred_model_ids = [
        resident_model_id,
        Some(astronomical_config::BUILTIN_DEFAULT_DECISION_MODEL_ID),
    ];
    for preferred_model_id in preferred_model_ids.into_iter().flatten() {
        for installed_model_id in &installed_decision_model_ids {
            if installed_model_id == preferred_model_id {
                return Ok(installed_model_id.clone());
            }
        }
    }
    match installed_decision_model_ids.len() {
        0 => Err(DecisionsModelResolutionError::NoDecisionModelInstalled),
        1 => Ok(installed_decision_model_ids.remove(0)),
        _ => {
            installed_decision_model_ids.sort();
            Err(
                DecisionsModelResolutionError::SeveralDecisionModelsInstalled {
                    installed_model_ids: installed_decision_model_ids,
                },
            )
        }
    }
}

/// Builds the worker command for a request whose questions are well formed.
///
/// The questions are parsed here so a malformed request is refused before it
/// can trigger a model swap; the worker parses them again on its side of the
/// trust boundary.
pub(crate) fn build_decisions_command(
    request_identifier: u64,
    resolved_model_id: &str,
    state: OrderedJsonValue,
    questions: OrderedJsonValue,
) -> Result<DecisionsCommand, QuestionDefinitionError> {
    parse_system_one_questions(&questions)?;
    Ok(DecisionsCommand {
        request_id: RequestId::new(request_identifier),
        model: resolved_model_id.to_owned(),
        state,
        questions,
    })
}

#[cfg(test)]
#[path = "decisions_service_tests.rs"]
mod decisions_service_tests;
