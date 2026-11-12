//! Speech logic shared by the REST endpoint and the daemon IPC service:
//! which model answers a request, and the command the worker receives.
//!
//! Both surfaces accept the same request and must agree on which model it
//! reaches and which seed it uses, so those choices are made here once and
//! each surface only translates the outcome into its own response.

use astronomical_config::{DiscoveredModel, ModelCapabilities};
use astronomical_ipc_protocol::{RequestId, SpeechCommand};
use astronomical_speech_contract::{NATURAL_PACE_ONLY_REASON, unknown_voice_reason};
use thiserror::Error;

use crate::serving::helper_model_resolution::{
    HelperModelKind, HelperModelResolutionError, resolve_helper_model,
};
use crate::serving::request_seed::generated_seed;

const SPEECH_MODEL_KIND: HelperModelKind = HelperModelKind {
    is_of_kind: is_speech_model,
    builtin_default_model_id: astronomical_config::BUILTIN_DEFAULT_SPEECH_MODEL_ID,
};

/// Why no model can be chosen for a speech request.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum SpeechModelResolutionError {
    #[error("model is not available to the local worker")]
    ModelNotFound,
    #[error("the requested model does not speak")]
    ModelCannotSpeak,
    #[error("no speech model is installed; download one from the Models page")]
    NoSpeechModelInstalled,
    #[error(
        "more than one speech model is installed ({}); name one with `model`",
        installed_model_ids.join(", ")
    )]
    SeveralSpeechModelsInstalled { installed_model_ids: Vec<String> },
}

impl From<HelperModelResolutionError> for SpeechModelResolutionError {
    fn from(resolution_error: HelperModelResolutionError) -> Self {
        match resolution_error {
            HelperModelResolutionError::ModelNotFound => Self::ModelNotFound,
            HelperModelResolutionError::ModelOfAnotherKind => Self::ModelCannotSpeak,
            HelperModelResolutionError::NoneInstalled => Self::NoSpeechModelInstalled,
            HelperModelResolutionError::SeveralInstalled {
                installed_model_ids,
            } => Self::SeveralSpeechModelsInstalled {
                installed_model_ids,
            },
        }
    }
}

/// Chooses the model that speaks a request: the named one, else the
/// resident speech model, the built-in default, or the only installed one.
pub(crate) fn resolve_speech_model(
    discovered_models: &[DiscoveredModel],
    resident_model_id: Option<&str>,
    requested_model: Option<&str>,
) -> Result<String, SpeechModelResolutionError> {
    resolve_helper_model(
        discovered_models,
        resident_model_id,
        requested_model,
        &SPEECH_MODEL_KIND,
    )
    .map_err(SpeechModelResolutionError::from)
}

/// The voices and pace a discovered speech model offers, to check a request
/// against before it can trigger a model swap.
pub(crate) fn speech_capabilities_of<'model>(
    discovered_models: &'model [DiscoveredModel],
    model_id: &str,
) -> Option<&'model astronomical_config::SpeechModelCapabilities> {
    discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_id == model_id)
        .and_then(|discovered_model| match &discovered_model.capabilities {
            ModelCapabilities::Speech(speech_capabilities) => Some(speech_capabilities),
            _ => None,
        })
}

/// Why a voice or pace was refused before the model was made resident.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum VoiceOrPaceRefusal {
    UnknownVoice { reason: String },
    FixedPace,
}

impl VoiceOrPaceRefusal {
    /// The words the caller reads, the same ones the engine would use.
    pub(crate) fn reason(&self) -> String {
        match self {
            Self::UnknownVoice { reason } => reason.clone(),
            Self::FixedPace => NATURAL_PACE_ONLY_REASON.to_owned(),
        }
    }
}

/// Refuses a voice the chosen model lacks, or a pace it cannot change,
/// before the request can make that model resident; the engine checks the
/// same again once loaded. A model discovery knows nothing more about passes.
pub(crate) fn refuse_voice_or_pace_before_swap(
    speech_capabilities: Option<&astronomical_config::SpeechModelCapabilities>,
    voice_name: Option<&str>,
    speed: f32,
) -> Result<(), VoiceOrPaceRefusal> {
    let Some(speech_capabilities) = speech_capabilities else {
        return Ok(());
    };
    if let Some(voice_name) = voice_name
        && !speech_capabilities
            .voice_names
            .iter()
            .any(|offered_voice_name| offered_voice_name == voice_name)
    {
        return Err(VoiceOrPaceRefusal::UnknownVoice {
            reason: unknown_voice_reason(voice_name, &speech_capabilities.voice_names),
        });
    }
    if !speech_capabilities.supports_speed && speed != 1.0 {
        return Err(VoiceOrPaceRefusal::FixedPace);
    }
    Ok(())
}

/// What a caller asked to be spoken, before a model was chosen.
pub(crate) struct SpeechAsked {
    pub(crate) text: String,
    pub(crate) voice_name: Option<String>,
    pub(crate) speed: f32,
    pub(crate) seed: Option<u64>,
}

/// The worker command for a request. A caller who named no seed gets one the
/// daemon picks, which the result reports, so any audio can be reproduced.
pub(crate) fn build_speech_command(
    request_identifier: u64,
    resolved_model_id: &str,
    speech_asked: SpeechAsked,
) -> SpeechCommand {
    SpeechCommand {
        request_id: RequestId::new(request_identifier),
        model: resolved_model_id.to_owned(),
        text: speech_asked.text,
        voice_name: speech_asked.voice_name,
        speed: speech_asked.speed,
        seed: speech_asked
            .seed
            .unwrap_or_else(|| generated_seed(request_identifier)),
    }
}

fn is_speech_model(model_capabilities: &ModelCapabilities) -> bool {
    matches!(model_capabilities, ModelCapabilities::Speech(_))
}

#[cfg(test)]
#[path = "speech_service_tests.rs"]
mod speech_service_tests;
