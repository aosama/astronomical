//! End-to-end acceptance for the configured Ornith 1.5 6-bit artifact.
//!
//! This artifact's snapshot files resolve through the Hugging Face shared blob
//! store rather than through its own cache entry's `blobs/` directory, so the
//! journey is the regression guard for that cache layout: it fails if validation
//! rejects a legitimate shared blob and passes only when the production engine
//! both loads the artifact and generates tokens from it.

use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    PerformanceAttribution, PerformanceAttributionLog, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
};
use tokio::time::timeout;

const ORNITH_ACCEPTANCE_MODEL_ID: &str = "Ornith-1.5-35B-A3B-MLX-6bit";
const ORNITH_ACCEPTANCE_PHASE_NAME: &str = "ornith-shared-blob-artifact";
const ORNITH_ACCEPTANCE_MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;
const ORNITH_ACCEPTANCE_OUTPUT_TOKEN_COUNT: u16 = 8;
const ORNITH_ACCEPTANCE_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 16;
const ORNITH_ACCEPTANCE_PROMPT_CHARACTER_COUNT: usize = 1_024;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

/// Loading a 35-billion-parameter artifact from disk is the repository's
/// documented exception to the 120-second test boundary. The bound exists only
/// so a wedged journey cannot hold wired GPU memory indefinitely.
const ORNITH_ACCEPTANCE_TIMEOUT_SECONDS: u64 = 900;

#[tokio::test]
#[ignore = "loads and generates with the configured Ornith 1.5 artifact through the production engine"]
async fn should_load_the_configured_ornith_artifact_through_the_production_engine() {
    timeout(
        Duration::from_secs(ORNITH_ACCEPTANCE_TIMEOUT_SECONDS),
        load_and_generate_with_configured_ornith_artifact(),
    )
    .await
    .expect("the configured Ornith journey must finish inside its documented real-model bound");
}

async fn load_and_generate_with_configured_ornith_artifact() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let development_config =
        astronomical_config::AstronomicalConfig::load_from_development_location()
            .expect("the Development configuration should load for the Ornith journey");
    let model_directory = crate::common::configured_discovered_model_by_id(
        &development_config,
        ORNITH_ACCEPTANCE_MODEL_ID,
    )
    .model_directory;

    eprintln!("[{ORNITH_ACCEPTANCE_PHASE_NAME}] status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, ORNITH_ACCEPTANCE_MAXIMUM_OUTPUT_TOKENS)
        .unwrap_or_else(|validation_error| {
            panic!(
                "[{ORNITH_ACCEPTANCE_PHASE_NAME}] the configured Ornith artifact should pass \
                 production validation: {validation_error}"
            )
        });
    let ornith_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured Ornith tokenizer should load from validated model metadata");
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&ornith_tokenizer);
    let end_of_sequence_token_ids = validated_artifact
        .config()
        .end_of_sequence_token_ids()
        .to_vec();

    let performance_attribution_directory = tempfile::tempdir()
        .expect("the Ornith journey should create a performance-attribution directory");
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let mut ornith_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(
            ORNITH_ACCEPTANCE_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
        )
        .expect("the Ornith journey prompt-processing chunk size should be valid"),
        crate::serving_acceptance::support::IMAGE_PAD_TOKEN_ID,
        model_directory,
        crate::common::standard_worker_chunking_configuration(),
        true,
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(
            &performance_attribution_directory
                .path()
                .join("performance-attribution.jsonl"),
            true,
        )
        .expect("the Ornith journey performance-attribution log should open"),
    )
    .expect("the configured Ornith engine settings should be valid");

    eprintln!("[{ORNITH_ACCEPTANCE_PHASE_NAME}] status=progress phase=model_load");
    crate::serving_acceptance::support::performance_attribution::load_engine_with_progress(
        &mut ornith_engine,
        ORNITH_ACCEPTANCE_PHASE_NAME,
    )
    .await;

    eprintln!("[{ORNITH_ACCEPTANCE_PHASE_NAME}] status=progress phase=romeo_and_juliet_generation");
    let generated_token_ids =
        crate::serving_acceptance::support::performance_attribution::run_attributed_generation(
            &mut ornith_engine,
            RequestId::new(96_001),
            &prompt_token_ids,
            ORNITH_ACCEPTANCE_PHASE_NAME,
            ORNITH_ACCEPTANCE_OUTPUT_TOKEN_COUNT,
            &end_of_sequence_token_ids,
        )
        .await;

    let model_vocabulary_size = ornith_tokenizer.model_vocabulary_size();
    assert!(
        !generated_token_ids.is_empty(),
        "the loaded Ornith artifact should generate at least one token"
    );
    for generated_token_id in &generated_token_ids {
        assert!(
            *generated_token_id < model_vocabulary_size,
            "generated token id {generated_token_id} must be within vocabulary size {model_vocabulary_size}"
        );
    }
    eprintln!(
        "[{ORNITH_ACCEPTANCE_PHASE_NAME}] status=success prompt_tokens={} output_tokens={}",
        prompt_token_ids.len(),
        generated_token_ids.len()
    );
}

/// Renders and encodes a bounded Romeo and Juliet prompt through the artifact's
/// own tokenizer, so the prompt proves the shared-blob tokenizer serves as well.
fn romeo_and_juliet_prompt_token_ids(ornith_tokenizer: &Qwen3_5Tokenizer) -> Vec<u32> {
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(ORNITH_ACCEPTANCE_PROMPT_CHARACTER_COUNT)
        .collect::<String>();
    let rendered_prompt = format!(
        "<|im_start|>user\nUse this Romeo and Juliet source:\n{source_excerpt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let prompt_token_ids = ornith_tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the bounded rendered Romeo and Juliet prompt should encode");
    assert!(
        !prompt_token_ids.is_empty(),
        "the rendered Romeo and Juliet prompt should produce tokens"
    );
    prompt_token_ids
}
