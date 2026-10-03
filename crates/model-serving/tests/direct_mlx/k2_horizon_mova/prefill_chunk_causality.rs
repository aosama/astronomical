//! End-to-end chunked-prefill journey: the same seeded request must decode
//! identically whether the prompt prefills in one chunk or several.
//!
//! Sampled tokens can hide a small chunking divergence behind near-tie
//! sampling, so the strict regression guard for chunked-prefill causality is
//! the in-crate logits probe
//! (`k2_horizon_mova::model::prefill_causality_probe`); this journey proves
//! the engine path end to end.

use std::time::Duration;

use astronomical_ipc_protocol::{ChatMessage, ChatToolChoice, RequestId};
use astronomical_model_serving::{
    GeneratedToken, K2HorizonMoVAConfig, K2HorizonMoVAInferenceRequest,
    K2HorizonMoVAPromptRenderer, K2HorizonMoVAServingSettings, K2HorizonMoVATokenizer,
    MlxInferenceExecution, initialize_k2_horizon_mova_execution_with_serving_settings,
};
use tokio::time::timeout;

const ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const CONTINUATION_TOKEN_COUNT: u32 = 8;
const SINGLE_CHUNK_TOKEN_COUNT: u32 = 256;
const MULTI_CHUNK_TOKEN_COUNT: u32 = 64;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

/// The prompt must span at least three multi-chunk-config chunks so two
/// multi-token prefill chunks run at a nonzero rope offset, and must stay
/// within one single-chunk-config chunk so the reference leg is one call.
const PROMPT_TOKEN_COUNT: usize = 200;

#[tokio::test]
#[ignore = "requires model_directories to discover a stacked affine K2 Horizon MoVA artifact"]
async fn should_generate_identical_continuation_regardless_of_prefill_chunking() {
    eprintln!(
        "[k2-prefill-causality] status=start timeout_seconds={}",
        ACCEPTANCE_TIMEOUT.as_secs()
    );
    timeout(ACCEPTANCE_TIMEOUT, run_prefill_chunk_causality_acceptance())
        .await
        .expect("the prefill chunk causality acceptance should finish within 115 seconds");
}

async fn run_prefill_chunk_causality_acceptance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_installed_model_directory_by_id(
        crate::common::k2_horizon_mova_model_id(),
    );
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&model_directory);
    eprintln!(
        "[k2-prefill-causality] status=prompt-ready prompt_tokens={}",
        prompt_token_ids.len()
    );

    // One prefill chunk spanning the whole prompt is causally correct by
    // construction: the causal kernel aligns its mask diagonal when the cache
    // is empty, so this leg is the reference any chunked prefill must
    // reproduce with the same seed.
    let single_chunk_continuation = generate_continuation(
        &model_directory,
        &memory_limits,
        &prompt_token_ids,
        SINGLE_CHUNK_TOKEN_COUNT,
    )
    .await;
    eprintln!("[k2-prefill-causality] single_chunk_continuation={single_chunk_continuation:?}");

    let multi_chunk_continuation = generate_continuation(
        &model_directory,
        &memory_limits,
        &prompt_token_ids,
        MULTI_CHUNK_TOKEN_COUNT,
    )
    .await;
    eprintln!("[k2-prefill-causality] multi_chunk_continuation={multi_chunk_continuation:?}");
    assert_eq!(
        single_chunk_continuation, multi_chunk_continuation,
        "the same seeded request must decode identically whether the prompt \
         prefills in one chunk or in {MULTI_CHUNK_TOKEN_COUNT}-token chunks; a divergence \
         means a multi-token continuation chunk attends tokens after itself"
    );
    eprintln!("[k2-prefill-causality] status=success");
}

async fn generate_continuation(
    model_directory: &std::path::Path,
    memory_limits: &astronomical_runtime_integration::MlxMemoryLimits,
    prompt_token_ids: &[u32],
    prompt_processing_chunk_tokens: u32,
) -> Vec<u32> {
    let mut serving_settings = K2HorizonMoVAServingSettings::default_fixed();
    serving_settings.prompt_processing_chunk_tokens = prompt_processing_chunk_tokens;
    let (_processor, mut execution) = initialize_k2_horizon_mova_execution_with_serving_settings(
        model_directory,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        false,
        serving_settings,
    )
    .expect("the installed K2 artifact should start");
    execution
        .load()
        .expect("K2 weights should load on GPU for the causality journey");
    let request = K2HorizonMoVAInferenceRequest::new(
        prompt_token_ids.to_vec(),
        CONTINUATION_TOKEN_COUNT,
        100,
        950,
        Some(1),
    );
    execution
        .start_generation(request)
        .expect("the seeded generation should start");
    let mut continuation_token_ids = Vec::new();
    for decode_attempt in 0..(CONTINUATION_TOKEN_COUNT * 4) {
        match execution
            .decode_next_token(RequestId::new(u64::from(decode_attempt) + 1))
            .expect("generation should advance")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => eprintln!(
                "[k2-prefill-causality] chunk={prompt_processing_chunk_tokens} prefill_processed_tokens={processed_token_count}"
            ),
            GeneratedToken::TokenId { token_id, .. } => {
                continuation_token_ids.push(token_id);
            }
            GeneratedToken::EndOfSequence => break,
            other => panic!("unexpected generation event: {other:?}"),
        }
    }
    assert_eq!(
        continuation_token_ids.len() as u32,
        CONTINUATION_TOKEN_COUNT,
        "the journey should decode {CONTINUATION_TOKEN_COUNT} tokens"
    );
    continuation_token_ids
}

fn romeo_and_juliet_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let model_config = K2HorizonMoVAConfig::from_json_bytes(
        &std::fs::read(model_directory.join("config.json"))
            .expect("the installed K2 config should be readable"),
    )
    .expect("the installed K2 config should parse");
    let tokenizer = K2HorizonMoVATokenizer::from_json_bytes(
        &std::fs::read(model_directory.join("tokenizer.json"))
            .expect("the installed K2 tokenizer should be readable"),
        &model_config,
    )
    .expect("the installed K2 tokenizer should load");
    let prompt_renderer = K2HorizonMoVAPromptRenderer::new();
    let rendered_prompt = prompt_renderer.render(
        &[ChatMessage::User {
            content: format!(
                "Use the supplied Romeo and Juliet source as the only source.\n\n{}",
                ROMEO_AND_JULIET_SOURCE
            ),
            images: Vec::new(),
        }],
        &[],
        &ChatToolChoice::None,
    );
    let prompt_token_ids = tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the Romeo and Juliet prompt should encode");
    assert!(
        prompt_token_ids.len() >= PROMPT_TOKEN_COUNT,
        "the Romeo and Juliet fixture should reach {PROMPT_TOKEN_COUNT} K2 prompt tokens"
    );
    prompt_token_ids[..PROMPT_TOKEN_COUNT].to_vec()
}
