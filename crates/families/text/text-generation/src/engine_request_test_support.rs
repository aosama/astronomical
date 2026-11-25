//! A minimal engine request for unit tests: an empty decoder state from a
//! small layout of three linear-attention layers and one full-attention
//! layer, disabled attribution, greedy sampling and no thinking budget, so
//! request logic is testable without MLX work.

use ipc_protocol::{RequestId, WorkerPromptWorkReuse};

use crate::astro_decoder_state_cache::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheTensorDtype, AstroDecoderStateCacheTensorLayout,
};
use crate::decoder_state::RequestDecoderStateStack;
use crate::decoder_state::tensor_roles::{
    ATTENTION_KEYS_TENSOR_ROLE, ATTENTION_VALUES_TENSOR_ROLE, LINEAR_CONVOLUTION_TENSOR_ROLE,
    LINEAR_RECURRENCE_TENSOR_ROLE,
};
use crate::engine_request::TextEngineRequest;
use crate::{PerformanceAttribution, TextSamplingStrategy, ThinkingBudgetState};

/// The full-attention key/value slab granularity of the test layout.
pub const TEST_KEY_VALUE_SLAB_TOKENS: usize = 256;

/// Three linear-attention layers then one full-attention layer, the order
/// the decoder stacks of the supported text families repeat.
const TEST_LAYER_IS_FULL_ATTENTION: [bool; 4] = [false, false, false, true];

/// A request with `engine_extension`, at position zero with no prompt.
pub fn test_engine_request<RequestExtension>(
    engine_extension: RequestExtension,
) -> TextEngineRequest<RequestExtension> {
    TextEngineRequest {
        request_decoder_state: empty_test_decoder_state(),
        generated_token_count: 0,
        input_token_ids: Vec::new(),
        last_restored_astro_ssd_prompt_cache_block_key: None,
        can_use_astro_ssd_prompt_cache: false,
        maximum_output_tokens: 1,
        astro_ssd_prompt_cache_block_causal_inputs: Vec::new(),
        next_position_tokens: 0,
        pending_generated_token: None,
        prefill_cursor: 0,
        random_state: None,
        request_id: RequestId::new(1),
        sampling_strategy: TextSamplingStrategy::HighestLogit,
        repetition_penalty_thousandths: 1_000,
        penalized_token_ids: Vec::new(),
        visual_embeddings: None,
        consumed_visual_embedding_count: 0,
        has_visual_inputs: false,
        image_pad_token_id: 0,
        thinking_budget_state: ThinkingBudgetState::new(false, None, Vec::new(), Vec::new())
            .expect("a request outside thinking needs no forced transition"),
        performance_attribution: PerformanceAttribution::disabled(),
        prompt_work_reuse: WorkerPromptWorkReuse::default(),
        astro_ssd_prompt_cache_diagnostics: None,
        generation_residency_preparation_attempted: false,
        first_decode_forward_elapsed_millis: None,
        outgrown_attention_slab_reclaim: Default::default(),
        generation_preparation_announced: false,
        structured_generation: None,
        engine_extension,
    }
}

/// A four-layer `astro_decoder_state_cache` layout with BFloat16 activations:
/// small enough to allocate in a unit test, with both layer kinds present.
pub fn test_astro_decoder_state_cache_layout() -> AstroDecoderStateCacheLayout {
    let activation_dtype = AstroDecoderStateCacheTensorDtype::BFloat16;
    let mut decoder_layer_layouts = Vec::new();
    for layer_is_full_attention in TEST_LAYER_IS_FULL_ATTENTION {
        decoder_layer_layouts.push(if layer_is_full_attention {
            AstroDecoderStateCacheLayerLayout::append_only_attention(
                AstroDecoderStateCacheTensorLayout::sequence(
                    ATTENTION_KEYS_TENSOR_ROLE,
                    activation_dtype,
                    vec![1, 2, 0, 16],
                    2,
                ),
                AstroDecoderStateCacheTensorLayout::sequence(
                    ATTENTION_VALUES_TENSOR_ROLE,
                    activation_dtype,
                    vec![1, 2, 0, 16],
                    2,
                ),
                TEST_KEY_VALUE_SLAB_TOKENS,
            )
        } else {
            AstroDecoderStateCacheLayerLayout::composite(vec![
                AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                    AstroDecoderStateCacheTensorLayout::fixed(
                        LINEAR_CONVOLUTION_TENSOR_ROLE,
                        activation_dtype,
                        vec![1, 3, 48],
                    ),
                ),
                AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                    AstroDecoderStateCacheTensorLayout::fixed(
                        LINEAR_RECURRENCE_TENSOR_ROLE,
                        AstroDecoderStateCacheTensorDtype::Float32,
                        vec![1, 4, 8, 8],
                    ),
                ),
            ])
        });
    }
    AstroDecoderStateCacheLayout::new(decoder_layer_layouts)
        .expect("the test layers form a valid astro-decoder-state-cache layout")
}

fn empty_test_decoder_state() -> RequestDecoderStateStack {
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout(
        &test_astro_decoder_state_cache_layout(),
    )
    .expect("the test layout builds an empty decoder state")
}
