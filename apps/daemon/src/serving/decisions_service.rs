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

use crate::serving::helper_model_resolution::{
    HelperModelKind, HelperModelResolutionError, resolve_helper_model,
};

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

/// Chooses the model that answers a decisions request, by the rule every
/// helper kind shares (`helper_model_resolution`): the named model, else the
/// resident decision model, the built-in default, or the only installed one.
pub(crate) fn resolve_decisions_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
    requested_model: Option<&str>,
) -> Result<String, DecisionsModelResolutionError> {
    resolve_helper_model(
        discovered_models,
        resident_model_id,
        requested_model,
        &DECISION_MODEL_KIND,
    )
    .map_err(DecisionsModelResolutionError::from)
}

const DECISION_MODEL_KIND: HelperModelKind = HelperModelKind {
    is_of_kind: is_decision_model,
    builtin_default_model_id: astronomical_config::BUILTIN_DEFAULT_DECISION_MODEL_ID,
};

fn is_decision_model(model_capabilities: &ModelCapabilities) -> bool {
    matches!(model_capabilities, ModelCapabilities::Decisions(_))
}

impl From<HelperModelResolutionError> for DecisionsModelResolutionError {
    fn from(resolution_error: HelperModelResolutionError) -> Self {
        match resolution_error {
            HelperModelResolutionError::ModelNotFound => Self::ModelNotFound,
            HelperModelResolutionError::ModelOfAnotherKind => Self::ModelCannotDecide,
            HelperModelResolutionError::NoneInstalled => Self::NoDecisionModelInstalled,
            HelperModelResolutionError::SeveralInstalled {
                installed_model_ids,
            } => Self::SeveralDecisionModelsInstalled {
                installed_model_ids,
            },
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
