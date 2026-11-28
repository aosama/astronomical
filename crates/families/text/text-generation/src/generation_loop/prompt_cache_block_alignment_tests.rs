use engine_core::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheTensorDtype, AstroDecoderStateCacheTensorLayout,
    AstroSsdPromptCacheModelContract,
};

use ipc_protocol::worker_chunking_test_support::worker_chunking_with_prompt_chunks;

use super::{
    PromptCacheContractInputs, PromptChunkSizes, chunk_aligned_block_token_count,
    resolve_chunk_aligned_contract,
};

const ONE_GIGABYTE_BYTES: u64 = 1_073_741_824;

/// One attention layer whose block length derives to its capacity growth,
/// with no recurrent state.
fn attention_contract_inputs(
    capacity_growth_tokens: usize,
    maximum_context_token_count: usize,
) -> PromptCacheContractInputs {
    PromptCacheContractInputs {
        model_id: "test-model".to_owned(),
        model_revision: "test-revision".to_owned(),
        astro_decoder_state_cache_layout: AstroDecoderStateCacheLayout::new(vec![
            AstroDecoderStateCacheLayerLayout::append_only_attention(
                attention_tensor_layout("attention.keys"),
                attention_tensor_layout("attention.values"),
                capacity_growth_tokens,
            ),
        ])
        .expect("the single-layer test layout validates"),
        maximum_context_token_count,
        active_memory_limit_bytes: ONE_GIGABYTE_BYTES,
        global_quota_bytes: 64 * ONE_GIGABYTE_BYTES,
        common_prefix_stride_blocks: 4,
    }
}

fn attention_tensor_layout(role_name: &str) -> AstroDecoderStateCacheTensorLayout {
    AstroDecoderStateCacheTensorLayout::sequence(
        role_name,
        AstroDecoderStateCacheTensorDtype::Float16,
        vec![1, 1, 0, 1],
        2,
    )
}

const ORNITH_PROMPT_CHUNK_SIZES: PromptChunkSizes = PromptChunkSizes {
    resident_token_count: 2_048,
    ssd_streaming_token_count: 4_096,
};

#[test]
fn should_resolve_a_derived_contract_at_whole_prompt_chunks() {
    let model_contract = resolve_chunk_aligned_contract(
        AstroSsdPromptCacheModelContract::resolve,
        &attention_contract_inputs(256, 262_144),
        None,
        ORNITH_PROMPT_CHUNK_SIZES,
    )
    .expect("the aligned contract resolves");
    assert_eq!(model_contract.block_token_count(), 4_096);
}

#[test]
fn should_keep_a_configured_block_length_as_it_is() {
    let model_contract = resolve_chunk_aligned_contract(
        AstroSsdPromptCacheModelContract::resolve,
        &attention_contract_inputs(256, 262_144),
        Some(512),
        ORNITH_PROMPT_CHUNK_SIZES,
    )
    .expect("the configured contract resolves");
    assert_eq!(model_contract.block_token_count(), 512);
}

#[test]
fn should_keep_the_derived_block_when_the_contract_refuses_the_aligned_one() {
    // A 3,000-token persistence alignment refuses 4,096, which is not a whole
    // number of it; the derived 3,000 stays.
    let model_contract = resolve_chunk_aligned_contract(
        AstroSsdPromptCacheModelContract::resolve,
        &attention_contract_inputs(3_000, 262_144),
        None,
        ORNITH_PROMPT_CHUNK_SIZES,
    )
    .expect("the derived contract resolves");
    assert_eq!(model_contract.block_token_count(), 3_000);
}

#[test]
fn should_refuse_a_configured_block_the_contract_refuses() {
    let resolution = resolve_chunk_aligned_contract(
        AstroSsdPromptCacheModelContract::resolve,
        &attention_contract_inputs(256, 262_144),
        Some(300),
        ORNITH_PROMPT_CHUNK_SIZES,
    );
    assert!(
        resolution.is_err(),
        "300 is not a whole number of the 256-token alignment"
    );
}

#[test]
fn should_round_the_ornith_derived_block_up_to_two_resident_chunks() {
    assert_eq!(
        chunk_aligned_block_token_count(3_328, 2_048, 4_096, 262_144),
        Some(4_096)
    );
}

#[test]
fn should_keep_a_block_that_already_holds_whole_chunks() {
    assert_eq!(
        chunk_aligned_block_token_count(8_192, 2_048, 4_096, 262_144),
        None
    );
}

#[test]
fn should_keep_a_block_when_the_aligned_length_exceeds_the_context() {
    assert_eq!(
        chunk_aligned_block_token_count(3_328, 2_048, 4_096, 4_000),
        None
    );
}

#[test]
fn should_accept_an_aligned_length_exactly_at_the_context() {
    assert_eq!(
        chunk_aligned_block_token_count(3_328, 2_048, 4_096, 4_096),
        Some(4_096)
    );
}

#[test]
fn should_align_to_the_resident_chunk_when_the_two_chunk_sizes_do_not_nest() {
    assert_eq!(
        chunk_aligned_block_token_count(3_328, 2_048, 3_000, 262_144),
        Some(4_096)
    );
    assert_eq!(
        chunk_aligned_block_token_count(2_100, 2_048, 3_000, 262_144),
        Some(4_096)
    );
}

#[test]
fn should_align_to_the_resident_chunk_when_it_is_the_larger() {
    assert_eq!(
        chunk_aligned_block_token_count(1_280, 2_048, 512, 262_144),
        Some(2_048)
    );
}

#[test]
fn should_not_align_when_the_resident_chunk_size_is_zero() {
    assert_eq!(
        chunk_aligned_block_token_count(3_328, 0, 4_096, 262_144),
        None
    );
}

#[test]
fn should_keep_a_block_on_overflow() {
    assert_eq!(
        chunk_aligned_block_token_count(usize::MAX - 1, 2_048, 4_096, usize::MAX),
        None
    );
}

#[test]
fn should_take_the_settled_in_memory_chunk_and_the_streaming_chunk() {
    let chunk_sizes = PromptChunkSizes::of(&worker_chunking_with_prompt_chunks(Some(2_048), 4_096));

    assert_eq!(chunk_sizes.resident_token_count, 2_048);
    assert_eq!(chunk_sizes.ssd_streaming_token_count, 4_096);
}

#[test]
fn should_count_an_unsettled_automatic_chunk_as_the_streaming_chunk() {
    let chunk_sizes = PromptChunkSizes::of(&worker_chunking_with_prompt_chunks(None, 4_096));

    assert_eq!(chunk_sizes.resident_token_count, 4_096);
}
