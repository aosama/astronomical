//! Shared real-artifact setup for automatic expert-residency qualifications.
//!
//! Every journey uses the repository's Romeo and Juliet prompt fixture and the
//! production engine boundary. Memory ceilings and prompt lengths vary by test;
//! artifact discovery, tokenization, chunking, and generation remain identical.

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, GenerationFinalization, InferenceEngine, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};

pub(super) const CONSTRAINED_MLX_MEMORY_CEILING_BYTES: usize = 10_000_000_000;
pub(super) const RESIDENCY_LIFECYCLE_PROMPT_TOKEN_COUNT: usize = 256;
pub(super) const RESIDENCY_QUALIFICATION_PROMPT_TOKEN_COUNT: usize = 8_192;

pub(super) fn initialize_automatic_residency_tracing() {
    if let Err(test_tracing_initialization_error) = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init()
    {
        eprintln!(
            "[automatic-residency] status=progress tracing=already_initialized reason={test_tracing_initialization_error}"
        );
    }
}

pub(super) fn construct_automatic_residency_engine(
    model_directory: std::path::PathBuf,
    active_memory_limit_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    request_id: RequestId,
    required_prompt_token_count: usize,
) -> (Qwen3_5Engine, Vec<u32>, u32, usize) {
    assert!(
        model_directory.is_dir(),
        "the configured sparse checkpoint must be available"
    );
    eprintln!("[automatic-residency 0/5] status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the configured sparse artifact should validate");
    let model_id = validated_artifact.model_id().to_owned();
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer should expose validated control tokens");
    let think_end_token_id = tokenizer.think_end_token_id();
    let image_pad_token_id = tokenizer.image_pad_token_id();
    let representative_prompt = super::speculative_prefill_qualification_support::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        &model_directory,
        &model_id,
        request_id,
        required_prompt_token_count,
        2,
    );
    let total_context_token_count = representative_prompt
        .prompt_token_ids
        .len()
        .checked_add(2)
        .expect("the request context token count should fit usize");
    let context_memory_reservation_bytes = validated_artifact
        .config()
        .context_memory_reservation_bytes(total_context_token_count)
        .expect("the request context memory reservation should fit usize");
    let qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_artifact,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(2_048)
            .expect("the test prefill_chunck_tokens should be valid"),
        think_end_token_id,
        model_directory,
        crate::common::standard_worker_chunking_configuration(),
        false,
        true,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        astronomical_model_serving::PerformanceAttribution::disabled(),
        astronomical_model_serving::PerformanceAttributionLog::disabled(),
    )
    .expect("the automatic expert-residency engine settings should be valid");
    (
        qwen3_5_engine,
        representative_prompt.prompt_token_ids,
        image_pad_token_id,
        context_memory_reservation_bytes,
    )
}

pub(super) async fn serve_romeo_and_juliet_request(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
    progress_log_prefix: &str,
) -> GenerationFinalization {
    eprintln!(
        "[{progress_log_prefix} 2/5] status=progress phase=romeo_and_juliet_generation prompt_tokens={}",
        prompt_token_ids.len()
    );
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(request_id, prompt_token_ids, 2)
                .with_image_pad_token_id(image_pad_token_id),
        )
        .await
        .expect("the model should accept the Romeo and Juliet request");
    complete_started_romeo_and_juliet_request(qwen3_5_engine, request_id, progress_log_prefix).await
}

pub(super) async fn complete_started_romeo_and_juliet_request(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    progress_log_prefix: &str,
) -> GenerationFinalization {
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the model should advance the Romeo and Juliet request")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => eprintln!(
                "[{progress_log_prefix} 3/5] status=progress phase=prefill processed_tokens={processed_token_count}"
            ),
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
        }
    }
    qwen3_5_engine
        .cancel_generation(request_id)
        .await
        .expect("the Romeo and Juliet request should finalize cleanly")
}
