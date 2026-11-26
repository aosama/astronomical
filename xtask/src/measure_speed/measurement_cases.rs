//! What one run measures: each named model under the engine's current memory
//! limit, or the reference set's cases, each under its own limit, narrowed to
//! the named models when any are named.

use std::path::Path;

use super::measure_speed_options::MeasureSpeedOptions;
use super::reference_set::{ReferenceCase, load_reference_set};
use super::speed_prompts::USUAL_LONG_PROMPT_TOKENS;

/// Whether a case changes the engine's memory limit before loading its model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryLimitRequest {
    /// Measure under whatever limit the engine has: a named model outside the
    /// reference set.
    KeepEngineLimit,
    /// Serve the model under this limit in GB; `None` is the Mac's own.
    ServeUnder(Option<u64>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementCase {
    pub model_id: String,
    pub memory_limit_request: MemoryLimitRequest,
    pub long_prompt_tokens: u32,
}

pub fn measurement_cases(
    repository_root: &Path,
    measure_speed_options: &MeasureSpeedOptions,
) -> Result<Vec<MeasurementCase>, String> {
    if !measure_speed_options.should_use_reference_set {
        let mut named_cases = Vec::new();
        for model_id in &measure_speed_options.model_ids {
            named_cases.push(MeasurementCase {
                model_id: model_id.clone(),
                memory_limit_request: MemoryLimitRequest::KeepEngineLimit,
                long_prompt_tokens: USUAL_LONG_PROMPT_TOKENS,
            });
        }
        return Ok(named_cases);
    }
    reference_cases_for(
        load_reference_set(repository_root)?,
        &measure_speed_options.model_ids,
    )
}

/// Every reference case, or only those of the named models; naming a model
/// the set does not hold is refused, since its figures would not be
/// comparable with any release.
pub fn reference_cases_for(
    reference_cases: Vec<ReferenceCase>,
    named_model_ids: &[String],
) -> Result<Vec<MeasurementCase>, String> {
    if let Some(unknown_model_id) = named_model_ids.iter().find(|named_model_id| {
        !reference_cases
            .iter()
            .any(|reference_case| &reference_case.model_id == *named_model_id)
    }) {
        return Err(format!(
            "the reference set has no case for {unknown_model_id}"
        ));
    }
    let mut chosen_cases = Vec::new();
    for reference_case in reference_cases {
        if named_model_ids.is_empty() || named_model_ids.contains(&reference_case.model_id) {
            chosen_cases.push(MeasurementCase {
                long_prompt_tokens: reference_case
                    .long_prompt_tokens
                    .unwrap_or(USUAL_LONG_PROMPT_TOKENS),
                model_id: reference_case.model_id,
                memory_limit_request: MemoryLimitRequest::ServeUnder(
                    reference_case.maximum_mlx_memory_gb,
                ),
            });
        }
    }
    Ok(chosen_cases)
}

/// A reference case is recorded under the limit it asked for, so the limit
/// the engine served under must be that one: an adjusted or still-pending
/// limit would file the figures under another workload.
pub fn refuse_unless_served_as_requested(
    memory_limit_request: MemoryLimitRequest,
    served_maximum_mlx_memory_gb: Option<u64>,
) -> Result<(), String> {
    match memory_limit_request {
        MemoryLimitRequest::ServeUnder(requested_gb)
            if requested_gb != served_maximum_mlx_memory_gb =>
        {
            Err(format!(
                "the engine served under {served_maximum_mlx_memory_gb:?} GB, the case asked for {requested_gb:?} GB"
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "measurement_cases_tests.rs"]
mod measurement_cases_tests;
