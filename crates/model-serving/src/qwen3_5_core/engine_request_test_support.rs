//! A minimal Qwen3.5 engine request for unit tests: an empty decoder state
//! from the frozen Ornith layout, disabled attribution, greedy sampling and
//! no thinking budget, so request logic is testable without MLX work.

use astronomical_ipc_protocol::{RequestId, WorkerPromptWorkReuse};

use crate::decoder_cache::DecoderCacheTensorDtype;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::decoder::cache_layout::{
    Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout,
};
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config;
use crate::{PerformanceAttribution, Qwen3_5SamplingStrategy, Qwen3_5ThinkingBudgetState};

const KV_SLAB_TOKENS: usize = 256;

/// A request with `engine_extension`, at position zero with no prompt.
pub(crate) fn test_engine_request<RequestExtension>(
    engine_extension: RequestExtension,
) -> Qwen3_5EngineRequest<RequestExtension> {
    Qwen3_5EngineRequest {
        request_decoder_state: empty_ornith_decoder_state(),
        generated_token_count: 0,
        input_token_ids: Vec::new(),
        last_restored_persistent_prompt_cache_block_key: None,
        can_use_persistent_prompt_cache: false,
        maximum_output_tokens: 1,
        persistent_prompt_cache_block_causal_inputs: Vec::new(),
        next_position_tokens: 0,
        pending_generated_token: None,
        prefill_cursor: 0,
        random_state: None,
        request_id: RequestId::new(1),
        sampling_strategy: Qwen3_5SamplingStrategy::HighestLogit,
        repetition_penalty_thousandths: 1_000,
        penalized_token_ids: Vec::new(),
        visual_embeddings: None,
        consumed_visual_embedding_count: 0,
        has_visual_inputs: false,
        image_pad_token_id: 0,
        thinking_budget_state: Qwen3_5ThinkingBudgetState::new(false, None, Vec::new(), Vec::new())
            .expect("a request outside thinking needs no forced transition"),
        performance_attribution: PerformanceAttribution::disabled(),
        prompt_work_reuse: WorkerPromptWorkReuse::default(),
        persistent_prompt_cache_diagnostics: None,
        generation_residency_preparation_attempted: false,
        first_decode_forward_elapsed_millis: None,
        generation_preparation_announced: false,
        structured_generation: None,
        engine_extension,
    }
}

fn empty_ornith_decoder_state() -> RequestDecoderStateStack {
    let ornith_config = frozen_ornith_1_0_config();
    let mut decoder_layer_cache_dtypes = Vec::new();
    for decoder_layer_index in 0..ornith_config.layer_count() as usize {
        decoder_layer_cache_dtypes.push(
            if ornith_config.decoder_layer_is_full_attention(decoder_layer_index) {
                Qwen3_5DecoderLayerCacheDtypes::FullAttention {
                    keys: DecoderCacheTensorDtype::BFloat16,
                    values: DecoderCacheTensorDtype::BFloat16,
                }
            } else {
                Qwen3_5DecoderLayerCacheDtypes::LinearAttention {
                    convolution: DecoderCacheTensorDtype::BFloat16,
                }
            },
        );
    }
    let decoder_cache_layout =
        qwen3_5_decoder_cache_layout(&ornith_config, KV_SLAB_TOKENS, &decoder_layer_cache_dtypes)
            .expect("the frozen configuration builds a decoder-cache layout");
    RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
        .expect("an empty decoder state")
}
