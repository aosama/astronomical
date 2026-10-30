//! Beside-source unit tests for the pure units extracted out of the tail
//! capture step (`persistent_prompt_cache_tail_capture.rs`): the tail token
//! span geometry, the causal-input resolution, and the tail block identity
//! construction. The end-to-end best-effort capture journey stays with the
//! engine-level tests.

use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
    PersistentPromptCacheModelContract,
};

use super::{
    TailBlockCausalInputResolution, build_tail_block_key, prompt_tail_capture_span,
    resolve_tail_block_causal_input,
};

const FIXTURE_BLOCK_TOKEN_COUNT: usize = 4;
const FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT: usize = 128;

#[test]
fn should_capture_the_tail_span_when_the_chunk_ends_at_the_partial_tail_boundary() {
    let tail_capture_span = prompt_tail_capture_span(true, 6, FIXTURE_BLOCK_TOKEN_COUNT, 5)
        .expect("a chunk ending one token short of the prompt end should capture a tail");
    assert_eq!(tail_capture_span.tail_capture_end, 5);
    assert_eq!(tail_capture_span.tail_block_start, 4);
    assert_eq!(tail_capture_span.tail_block_index, 1);
    assert_eq!(
        tail_capture_span.persistent_prompt_cache_block_token_count,
        FIXTURE_BLOCK_TOKEN_COUNT
    );
}

#[test]
fn should_skip_tail_capture_when_the_request_cannot_use_the_persistent_prompt_cache() {
    assert_eq!(
        prompt_tail_capture_span(false, 6, FIXTURE_BLOCK_TOKEN_COUNT, 5),
        None
    );
}

#[test]
fn should_skip_tail_capture_when_the_chunk_ends_before_the_tail_boundary() {
    assert_eq!(
        prompt_tail_capture_span(true, 6, FIXTURE_BLOCK_TOKEN_COUNT, 4),
        None
    );
}

#[test]
fn should_skip_tail_capture_when_the_prompt_ends_exactly_on_a_block_boundary() {
    // A prompt that fills whole blocks leaves no partial tail to capture.
    assert_eq!(
        prompt_tail_capture_span(true, 8, FIXTURE_BLOCK_TOKEN_COUNT, 8),
        None
    );
}

#[test]
fn should_skip_tail_capture_when_the_clamped_capture_end_leaves_no_tail_tokens() {
    // A one-token prompt clamps the capture end to zero: the terminal forward
    // must process that token, and no partial tail can exist before it.
    assert_eq!(
        prompt_tail_capture_span(true, 1, FIXTURE_BLOCK_TOKEN_COUNT, 0),
        None
    );
}

#[test]
fn should_resolve_the_planned_causal_input_for_the_tail_block_index() {
    let planned_block_causal_inputs = vec![
        PersistentPromptCacheBlockCausalInput::empty(),
        PersistentPromptCacheBlockCausalInput::from_canonical_bytes(b"tail-capture-fixture"),
    ];
    match resolve_tail_block_causal_input(&planned_block_causal_inputs, 1) {
        TailBlockCausalInputResolution::Planned(tail_block_causal_input) => {
            assert_eq!(
                tail_block_causal_input,
                &PersistentPromptCacheBlockCausalInput::from_canonical_bytes(
                    b"tail-capture-fixture"
                )
            );
        }
        resolution => panic!("the plan entry should resolve, got {resolution:?}"),
    }
}

#[test]
fn should_report_a_missing_plan_entry_when_the_plan_does_not_cover_the_tail_block() {
    let planned_block_causal_inputs = vec![PersistentPromptCacheBlockCausalInput::empty()];
    assert!(matches!(
        resolve_tail_block_causal_input(&planned_block_causal_inputs, 1),
        TailBlockCausalInputResolution::MissingFromPlan
    ));
}

#[test]
fn should_fall_back_to_the_empty_causal_input_when_the_request_carries_no_plan() {
    let planned_block_causal_inputs = Vec::new();
    assert!(matches!(
        resolve_tail_block_causal_input(&planned_block_causal_inputs, 0),
        TailBlockCausalInputResolution::EmptyFallback
    ));
}

#[test]
fn should_build_a_root_tail_block_identity_when_the_request_restored_no_parent() {
    let persistent_prompt_cache_model_contract = synthetic_tail_capture_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let tail_tokens = vec![11_u32; block_token_count - 1];
    let tail_block_key = build_tail_block_key(
        &persistent_prompt_cache_model_contract,
        &tail_tokens,
        &PersistentPromptCacheBlockCausalInput::empty(),
        None,
    )
    .expect("a partial tail within one block should build a root identity");
    assert_eq!(tail_block_key.block_index(), 0);
    assert_eq!(tail_block_key.token_count(), block_token_count - 1);
}

#[test]
fn should_chain_the_tail_block_identity_under_the_restored_parent() {
    let persistent_prompt_cache_model_contract = synthetic_tail_capture_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let parent_block_key = PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract,
        &vec![7_u32; block_token_count],
    )
    .expect("the fixture parent block should build a root identity");
    let tail_tokens = vec![11_u32; block_token_count - 1];
    let tail_block_key = build_tail_block_key(
        &persistent_prompt_cache_model_contract,
        &tail_tokens,
        &PersistentPromptCacheBlockCausalInput::empty(),
        Some(&parent_block_key),
    )
    .expect("a partial tail under a restored parent should chain a child identity");
    assert_eq!(
        tail_block_key.block_index(),
        parent_block_key.block_index() + 1
    );
}

#[test]
fn should_reject_tail_tokens_exceeding_the_block_token_count() {
    let persistent_prompt_cache_model_contract = synthetic_tail_capture_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let oversized_tail_tokens = vec![11_u32; block_token_count + 1];
    assert!(matches!(
        build_tail_block_key(
            &persistent_prompt_cache_model_contract,
            &oversized_tail_tokens,
            &PersistentPromptCacheBlockCausalInput::empty(),
            None,
        ),
        Err(PersistentPromptCacheBlockKeyError::BlockTokenCountExceedsBlock { .. })
    ));
}

#[test]
fn should_reject_an_empty_tail_token_span() {
    let persistent_prompt_cache_model_contract = synthetic_tail_capture_contract();
    assert!(matches!(
        build_tail_block_key(
            &persistent_prompt_cache_model_contract,
            &[],
            &PersistentPromptCacheBlockCausalInput::empty(),
            None,
        ),
        Err(PersistentPromptCacheBlockKeyError::EmptyBlockTokens)
    ));
}

fn synthetic_tail_capture_contract() -> PersistentPromptCacheModelContract {
    // One append-only attention layer with four-token growth derives a
    // four-token block, so fixture tails stay small and readable.
    PersistentPromptCacheModelContract::resolve(
        "tail-capture-unit-fixture".to_owned(),
        "revision".to_owned(),
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            FIXTURE_BLOCK_TOKEN_COUNT,
        )])
        .expect("the synthetic sequence-only layout should be valid"),
        FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT,
        1_000_000,
        1_000_000,
        None,
        4,
    )
    .expect("the synthetic layout should resolve a storage contract")
}
