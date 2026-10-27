//! The quota-bounded persistent prompt-cache model-contract resolution shared
//! by every family that opens an SSD store.
//!
//! A global SSD quota can refuse a contract whose full-context block chain
//! does not fit. Resolution therefore halves the cacheable context until the
//! derived file geometry fits the quota, never dropping below one block of
//! cacheable context; every other validation failure is returned untouched.

use super::model_contract::PersistentPromptCacheModelContract;
use super::model_contract_error::PersistentPromptCacheModelContractError;
use crate::DecoderCacheLayout;

#[allow(clippy::too_many_arguments)]
/// Resolves the model contract, shrinking the cacheable context on quota
/// failures only.
pub(crate) fn resolve_quota_bounded_prompt_cache_contract(
    model_id: &str,
    model_revision: &str,
    decoder_cache_layout: DecoderCacheLayout,
    model_maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    global_ssd_quota_bytes: u64,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
) -> Result<PersistentPromptCacheModelContract, PersistentPromptCacheModelContractError> {
    let minimum_cacheable_context_token_count = configured_block_token_count.unwrap_or(1).max(1);
    let mut cacheable_context_token_count = model_maximum_context_token_count;
    loop {
        match PersistentPromptCacheModelContract::resolve(
            model_id.to_owned(),
            model_revision.to_owned(),
            decoder_cache_layout.clone(),
            cacheable_context_token_count,
            effective_mlx_memory_ceiling_bytes,
            global_ssd_quota_bytes,
            configured_block_token_count,
            common_prefix_checkpoint_stride_blocks,
        ) {
            Ok(model_contract) => return Ok(model_contract),
            Err(
                quota_error @ (PersistentPromptCacheModelContractError::ConfiguredBlockChainExceedsSsdQuota { .. }
                | PersistentPromptCacheModelContractError::BlockFilesExceedSsdQuota { .. }),
            ) => {
                let reduced_context_token_count = (cacheable_context_token_count / 2)
                    .max(minimum_cacheable_context_token_count);
                if reduced_context_token_count == cacheable_context_token_count {
                    return Err(quota_error);
                }
                cacheable_context_token_count = reduced_context_token_count;
            }
            Err(contract_error) => return Err(contract_error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DecoderCacheLayerLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
        PersistentPromptCacheModelContractError,
    };

    fn single_layer_layout() -> DecoderCacheLayout {
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 1, 0, 1],
                2,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 1, 0, 1],
                2,
            ),
            1,
        )])
        .expect("the single-layer test layout must validate")
    }

    #[test]
    fn should_shrink_the_cacheable_context_until_the_chain_fits_the_quota() {
        let model_contract = resolve_quota_bounded_prompt_cache_contract(
            "test-model",
            "test-revision",
            single_layer_layout(),
            1_048_576,
            1_073_741_824,
            // The full-context block chain far exceeds this quota, so
            // resolution must halve the cacheable context until it fits.
            65_536,
            Some(2_048),
            1,
        )
        .expect("a quota-feasible context must resolve");
        assert!(
            model_contract.maximum_context_token_count() < 1_048_576,
            "the quota must shrink the cacheable context below the model window"
        );
    }

    #[test]
    fn should_return_non_quota_failures_untouched() {
        let resolution_error = resolve_quota_bounded_prompt_cache_contract(
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
            PersistentPromptCacheModelContractError::EmptyModelId
        ));
    }
}
