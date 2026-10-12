use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::ChatGenerationSettings;

use crate::ResolvedRuntimeConfig;

/// Tracks which public settings were supplied so model policy changes only omissions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RequestGenerationSettingsPresence {
    pub maximum_output_tokens: bool,
    pub temperature: bool,
    pub top_p: bool,
}

/// Applies one canonical model's live request defaults without overriding client values.
pub(crate) fn apply_generation_defaults(
    reloadable_runtime_config: Option<&Arc<RwLock<ResolvedRuntimeConfig>>>,
    model_id: &str,
    settings_presence: RequestGenerationSettingsPresence,
    generation_settings: &mut ChatGenerationSettings,
) {
    let Some(reloadable_runtime_config) = reloadable_runtime_config else {
        return;
    };
    let Ok(resolved_runtime_config) = reloadable_runtime_config.read() else {
        return;
    };
    let Some(model_policy) = resolved_runtime_config.model_policy_catalog.get(model_id) else {
        return;
    };
    if !settings_presence.maximum_output_tokens {
        generation_settings.max_output_tokens =
            model_policy.generation_defaults.maximum_output_tokens;
    }
    if !settings_presence.temperature {
        generation_settings.temperature_thousandths =
            model_policy.generation_defaults.temperature_thousandths;
    }
    if !settings_presence.top_p {
        generation_settings.top_p_thousandths = model_policy.generation_defaults.top_p_thousandths;
    }
}
