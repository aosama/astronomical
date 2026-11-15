//! The quota-bounded persistent astro SSD prompt cache model-contract resolution shared
//! by every family that opens an SSD store.
//!
//! A global SSD quota can refuse a contract whose full-context block chain
//! does not fit. Resolution therefore halves the context the `astro_ssd_prompt_cache`
//! covers until the derived file geometry fits the quota, never dropping below one block of
//! that context; every other validation failure is returned untouched.

use super::model_contract::AstroSsdPromptCacheModelContract;
use super::model_contract_error::AstroSsdPromptCacheModelContractError;
use crate::AstroDecoderStateCacheLayout;

#[allow(clippy::too_many_arguments)]
/// Resolves the model contract, shrinking the context the `astro_ssd_prompt_cache` covers on quota
/// failures only.
pub(crate) fn resolve_quota_bounded_astro_ssd_prompt_cache_contract(
    model_id: &str,
    model_revision: &str,
    astro_decoder_state_cache_layout: AstroDecoderStateCacheLayout,
    model_maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    global_ssd_quota_bytes: u64,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
) -> Result<AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError> {
    let minimum_astro_ssd_prompt_cache_context_token_count =
        configured_block_token_count.unwrap_or(1).max(1);
    let mut astro_ssd_prompt_cache_context_token_count = model_maximum_context_token_count;
    loop {
        match AstroSsdPromptCacheModelContract::resolve(
            model_id.to_owned(),
            model_revision.to_owned(),
            astro_decoder_state_cache_layout.clone(),
            astro_ssd_prompt_cache_context_token_count,
            effective_mlx_memory_ceiling_bytes,
            global_ssd_quota_bytes,
            configured_block_token_count,
            common_prefix_checkpoint_stride_blocks,
        ) {
            Ok(model_contract) => return Ok(model_contract),
            Err(
                quota_error @ (AstroSsdPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota { .. }
                | AstroSsdPromptCacheModelContractError::BlockFilesExceedSsdQuota { .. }),
            ) => {
                let reduced_context_token_count = (astro_ssd_prompt_cache_context_token_count / 2)
                    .max(minimum_astro_ssd_prompt_cache_context_token_count);
                if reduced_context_token_count == astro_ssd_prompt_cache_context_token_count {
                    return Err(quota_error);
                }
                astro_ssd_prompt_cache_context_token_count = reduced_context_token_count;
            }
            Err(contract_error) => return Err(contract_error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheTensorDtype,
        AstroDecoderStateCacheTensorLayout, AstroSsdPromptCacheModelContractError,
    };

    fn single_layer_layout() -> AstroDecoderStateCacheLayout {
        AstroDecoderStateCacheLayout::new(vec![
            AstroDecoderStateCacheLayerLayout::append_only_attention(
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.keys",
                    AstroDecoderStateCacheTensorDtype::Float16,
                    vec![1, 1, 0, 1],
                    2,
                ),
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.values",
                    AstroDecoderStateCacheTensorDtype::Float16,
                    vec![1, 1, 0, 1],
                    2,
                ),
                1,
            ),
        ])
        .expect("the single-layer test layout must validate")
    }

    #[test]
    fn should_shrink_the_astro_ssd_prompt_cache_context_until_the_chain_fits_the_quota() {
        let model_contract = resolve_quota_bounded_astro_ssd_prompt_cache_contract(
            "test-model",
            "test-revision",
            single_layer_layout(),
            1_048_576,
            1_073_741_824,
            // The full-context block chain far exceeds this quota, so
            // resolution must halve the `astro_ssd_prompt_cache` context until it fits.
            65_536,
            Some(2_048),
            1,
        )
        .expect("a quota-feasible context must resolve");
        assert!(
            model_contract.maximum_context_token_count() < 1_048_576,
            "the quota must shrink the astro_ssd_prompt_cache context below the model window"
        );
    }

    #[test]
    fn should_return_non_quota_failures_untouched() {
        let resolution_error = resolve_quota_bounded_astro_ssd_prompt_cache_contract(
            "",
            "test-revision",
            single_layer_layout(),
            4_096,
            1_073_741_824,
            1_073_741_824,
            None,
            1,
        )
        .expect_err("an empty model id must fail validation");
        assert!(matches!(
            resolution_error,
            AstroSsdPromptCacheModelContractError::EmptyModelId
        ));
    }
}
