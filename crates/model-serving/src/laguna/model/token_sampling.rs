//! Executes the resolved Laguna sampler on GPU logits.
//!
//! Highest-logit selection stays argmax. Sampled requests use the shared MLX categorical graph so
//! temperature, top-k, and top-p cannot be logged as effective and then ignored.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::laguna::LagunaSamplerConfig;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaExecutionError;
use super::model::{LagunaModel, last_token_vocabulary_logits};

impl LagunaModel {
    /// Samples one vocabulary index with temperature, optional top-k, and top-p.
    pub fn sampled_token_id(
        runtime: &MlxRuntime,
        logits: &MlxArray,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        top_k: Option<u16>,
        random_state: &mut MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<u32, LagunaExecutionError> {
        let last_token_logits = last_token_vocabulary_logits(runtime, logits)?;
        let vocabulary_size = last_token_logits.shape().first().copied().ok_or_else(|| {
            LagunaExecutionError::invalid_geometry("Laguna logits are missing a vocabulary axis")
        })?;
        let batched_logits = runtime.reshape(&last_token_logits, &[1, 1, vocabulary_size])?;
        let executed_top_k = top_k.unwrap_or(LagunaSamplerConfig::DEFAULT_SAMPLING_TOP_K);
        let sampled_token = performance_attribution.measure_operation(
            PerformanceOperation::TokenSamplingGraphConstruction,
            |_| {
                crate::gpu_token_sampling::build_sampled_token(
                    runtime,
                    &batched_logits,
                    temperature_thousandths,
                    top_p_thousandths,
                    Some(executed_top_k),
                    random_state,
                )
                .map_err(|sampling_error| LagunaExecutionError::RuntimeOperation {
                    description: sampling_error.to_string(),
                })
            },
        )?;
        let scalar_token = squeeze_to_scalar_token(runtime, &sampled_token)?;
        performance_attribution
            .measure_operation(
                PerformanceOperation::GeneratedTokenItemSynchronizationWait,
                |_| scalar_token.item_u32(),
            )
            .map_err(LagunaExecutionError::from)
    }
}

fn squeeze_to_scalar_token(
    runtime: &MlxRuntime,
    sampled_token: &MlxArray,
) -> Result<MlxArray, LagunaExecutionError> {
    if sampled_token.shape().is_empty() {
        return Ok(sampled_token.retain()?);
    }
    let element_count = sampled_token
        .shape()
        .iter()
        .try_fold(1_i32, |element_count, dimension| {
            element_count.checked_mul(*dimension)
        })
        .ok_or_else(|| LagunaExecutionError::invalid_geometry("sampled token shape overflowed"))?;
    if element_count != 1 {
        return Err(LagunaExecutionError::invalid_geometry(
            "sampled token must contain one vocabulary index",
        ));
    }
    Ok(runtime.reshape(sampled_token, &[])?)
}
