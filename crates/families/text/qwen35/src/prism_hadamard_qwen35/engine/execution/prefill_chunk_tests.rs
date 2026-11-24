use ipc_protocol::{RequestId, WorkerEvent};
use serial_test::serial;

use crate::prism_hadamard_qwen35::engine::fixture_engine_test_support::{
    FixtureAstroSsdPromptCache, LoadedFixtureEngine,
    loaded_fixture_engine_with_astro_ssd_prompt_cache,
};
use crate::qwen3_5_core::astro_ssd_prompt_cache_capture::Qwen3_5AstroSsdPromptCacheCapture;
use crate::{
    AstroSsdPromptCacheModelContract, EngineGenerationStart, GeneratedToken, MlxInferenceExecution,
    Qwen3_5InferenceRequest,
};
use engine_core::astro_ssd_prompt_cache::quota_bounded_contract::resolve_quota_bounded_astro_ssd_prompt_cache_contract;

/// Blocks of 256 tokens, the fixture layout's persistence alignment, read in
/// chunks of 300 so every chunk holds one block boundary before its end and
/// the folded-ternary forward records the boundary states itself.
const FIXTURE_ASTRO_SSD_PROMPT_CACHE: FixtureAstroSsdPromptCache = FixtureAstroSsdPromptCache {
    prompt_processing_chunk_tokens: 300,
    block_token_count: 256,
};

/// 600 prompt tokens: two whole blocks and an 88-token partial tail.
fn repeated_prompt_token_ids() -> Vec<u32> {
    (0..600_u32)
        .map(|token_position| 11 + token_position % 400)
        .collect()
}

fn start_greedy(
    fixture_engine: &mut LoadedFixtureEngine,
    request_id: u64,
) -> EngineGenerationStart {
    fixture_engine
        .execution
        .start_generation(Qwen3_5InferenceRequest::new_sampling(
            RequestId::new(request_id),
            repeated_prompt_token_ids(),
            2,
            0,
            1_000,
            None,
        ))
        .expect("the request should start")
}

/// Runs the request until it finalizes itself, returning the token count of
/// every prompt chunk it read.
fn run_to_completion(fixture_engine: &mut LoadedFixtureEngine, request_id: u64) -> Vec<u32> {
    let mut prompt_chunk_token_counts = Vec::new();
    for _ in 0..16 {
        match fixture_engine
            .execution
            .decode_next_token(RequestId::new(request_id))
            .expect("the step should succeed")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => prompt_chunk_token_counts.push(processed_token_count),
            GeneratedToken::TokenId {
                generation_finalization: Some(_),
                ..
            } => return prompt_chunk_token_counts,
            _ => {}
        }
    }
    panic!("the request never finalized itself");
}

#[test]
#[serial]
fn should_restore_a_repeated_prompt_up_to_its_partial_tail() {
    let mut fixture_engine =
        loaded_fixture_engine_with_astro_ssd_prompt_cache(&FIXTURE_ASTRO_SSD_PROMPT_CACHE);
    let cold_start = start_greedy(&mut fixture_engine, 91);
    assert_eq!(cold_start.restored_prompt_prefix_token_count(), 0);
    // The last chunk stops one token short so the 87-token tail up to the
    // final prompt token can be captured; the final token is read alone.
    assert_eq!(run_to_completion(&mut fixture_engine, 91), [300, 299, 1]);

    let warm_start = start_greedy(&mut fixture_engine, 92);

    // Both whole blocks and the partial tail are restored: everything but
    // the final prompt token, which the first reply token needs.
    assert_eq!(warm_start.restored_prompt_prefix_token_count(), 599);
    assert_eq!(run_to_completion(&mut fixture_engine, 92), [1]);
}

#[test]
#[serial]
fn should_open_the_store_with_the_quota_bounded_contract() {
    let fixture_engine =
        loaded_fixture_engine_with_astro_ssd_prompt_cache(&FIXTURE_ASTRO_SSD_PROMPT_CACHE);
    let execution_core = &fixture_engine.execution.execution_core;
    let model = execution_core.loaded_model().expect("the model is loaded");
    let expected_contract = resolve_quota_bounded_astro_ssd_prompt_cache_contract(
        model.pack().model_id(),
        model.pack().revision(),
        model.astro_decoder_state_cache_layout.clone(),
        model.pack().config().text().max_position_embeddings as usize,
        512 * 1024 * 1024,
        1_073_741_824,
        Some(256),
        1,
    )
    .expect("the fixture contract should resolve");
    let opened_contract: &AstroSsdPromptCacheModelContract = execution_core
        .astro_ssd_prompt_cache
        .as_ref()
        .expect("the store is open")
        .model_contract_ref();
    assert_eq!(opened_contract, &expected_contract);
    assert_eq!(
        execution_core
            .astro_ssd_prompt_cache_model_contract
            .as_ref(),
        Some(&expected_contract)
    );
}

#[test]
#[serial]
fn should_report_and_clear_the_store_through_the_shared_core() {
    let mut fixture_engine =
        loaded_fixture_engine_with_astro_ssd_prompt_cache(&FIXTURE_ASTRO_SSD_PROMPT_CACHE);
    start_greedy(&mut fixture_engine, 93);
    run_to_completion(&mut fixture_engine, 93);

    assert!(
        fixture_engine
            .execution
            .collect_astro_ssd_prompt_cache_stats()
            .expect("the statistics should be read")
            .is_some()
    );
    let Some(WorkerEvent::AstroSsdPromptCacheCleared { blocks_removed, .. }) = fixture_engine
        .execution
        .clear_astro_ssd_prompt_cache(None)
        .expect("the store should clear")
    else {
        panic!("an open store reports what it cleared");
    };
    assert!(blocks_removed >= 2, "both whole blocks were published");
    // Negative: with the store cleared, the same prompt starts cold.
    assert_eq!(
        start_greedy(&mut fixture_engine, 94).restored_prompt_prefix_token_count(),
        0
    );
}

#[test]
#[serial]
fn should_retry_a_blocked_publication_without_evicting_anything() {
    let mut fixture_engine =
        loaded_fixture_engine_with_astro_ssd_prompt_cache(&FIXTURE_ASTRO_SSD_PROMPT_CACHE);
    start_greedy(&mut fixture_engine, 95);
    let execution = &mut fixture_engine.execution;
    let mut active_generation = execution
        .execution_core
        .active_request
        .take()
        .expect("the request is active");
    let decoder_state_bytes = active_generation.request_decoder_state.payload_byte_count();
    let model = execution
        .execution_core
        .loaded_model()
        .expect("the model is loaded");
    execution
        .reclaim_before_publication_retry(model, &mut active_generation, 1_073_741_824)
        .expect("a folded pack has nothing to evict before the retry");
    assert_eq!(
        active_generation.request_decoder_state.payload_byte_count(),
        decoder_state_bytes
    );
}
