//! Request-owned Laguna token selection for one generation.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::laguna::{LagunaModel, LagunaSamplerConfig, LagunaSamplingStrategy};
use crate::sampling_seed::{current_time_millis_since_unix_epoch, resolve_sampling_seed};
use crate::{InferenceEngineError, PerformanceAttribution};

pub(super) fn random_state_for_strategy(
    runtime: &MlxRuntime,
    sampling_strategy: &LagunaSamplingStrategy,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    match sampling_strategy {
        LagunaSamplingStrategy::HighestLogit => Ok(None),
        LagunaSamplingStrategy::Sample(sampler_config) => {
            Ok(Some(random_state_for_sampler(runtime, sampler_config)?))
        }
    }
}

fn random_state_for_sampler(
    runtime: &MlxRuntime,
    sampler_config: &LagunaSamplerConfig,
) -> Result<MlxArray, InferenceEngineError> {
    if sampler_config.min_p_thousandths() > 0 {
        return Err(InferenceEngineError::InvalidRequest {
            reason: "Laguna min_p sampling is not implemented".to_owned(),
        });
    }
    let sampling_seed =
        resolve_sampling_seed(sampler_config.seed(), current_time_millis_since_unix_epoch);
    runtime
        .random_key(sampling_seed)
        .map_err(|runtime_error| InferenceEngineError::Fatal {
            reason: format!("Laguna sampling random key failed: {runtime_error:?}"),
        })
}

pub(super) fn select_next_token_id(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    sampling_strategy: &LagunaSamplingStrategy,
    random_state: &mut Option<MlxArray>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<u32, InferenceEngineError> {
    match sampling_strategy {
        LagunaSamplingStrategy::HighestLogit => {
            LagunaModel::highest_logit_token_id(runtime, logits, performance_attribution).map_err(
                |sampling_error| InferenceEngineError::Fatal {
                    reason: format!("Laguna highest-logit selection failed: {sampling_error:?}"),
                },
            )
        }
        LagunaSamplingStrategy::Sample(sampler_config) => {
            let sampling_random_state =
                random_state
                    .as_mut()
                    .ok_or_else(|| InferenceEngineError::Fatal {
                        reason: "sampled Laguna request lost its random state".to_owned(),
                    })?;
            LagunaModel::sampled_token_id(
                runtime,
                logits,
                sampler_config.temperature_thousandths(),
                sampler_config.top_p_thousandths(),
                sampler_config.top_k(),
                sampling_random_state,
                performance_attribution,
            )
            .map_err(|sampling_error| InferenceEngineError::Fatal {
                reason: format!("Laguna token sampling failed: {sampling_error:?}"),
            })
        }
    }
}

pub(super) fn log_executed_sampling(request_id: u64, sampling_strategy: &LagunaSamplingStrategy) {
    match sampling_strategy {
        LagunaSamplingStrategy::HighestLogit => {
            tracing::info!(
                request_id,
                sampling_selects_highest_logit = true,
                executed_temperature_thousandths = 0_u16,
                executed_top_k = Option::<u16>::None,
                executed_top_p_thousandths = 1_000_u16,
                "executed generation sampling"
            );
        }
        LagunaSamplingStrategy::Sample(sampler_config) => {
            tracing::info!(
                request_id,
                sampling_selects_highest_logit = false,
                executed_temperature_thousandths = sampler_config.temperature_thousandths(),
                executed_top_k = sampler_config.sampling_top_k(),
                executed_top_p_thousandths = sampler_config.top_p_thousandths(),
                "executed generation sampling"
            );
        }
    }
}
