use std::sync::{Arc, RwLock};

use crate::ResolvedRuntimeConfig;

/// Caps one IPC-representable request budget to the live configured ceiling.
pub(crate) fn cap_generation_output_tokens(
    reloadable_runtime_config: Option<&Arc<RwLock<ResolvedRuntimeConfig>>>,
    requested_output_tokens: u16,
) -> u16 {
    let Some(reloadable_runtime_config) = reloadable_runtime_config else {
        return requested_output_tokens;
    };
    let Ok(resolved_runtime_config) = reloadable_runtime_config.read() else {
        return requested_output_tokens;
    };
    let configured_output_tokens = match u16::try_from(resolved_runtime_config.max_output_tokens) {
        Ok(configured_output_tokens) if configured_output_tokens > 0 => configured_output_tokens,
        Ok(_) => 1,
        Err(_) => u16::MAX,
    };
    requested_output_tokens.min(configured_output_tokens)
}
