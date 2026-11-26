//! Engine construction and structural token checks shared by the installed
//! journeys that serve one short request through the text engine.

use std::path::Path;

use ipc_protocol::RequestId;
use model_serving::{
    GeneratedToken, InferenceEngine, PromptProcessingChunkSizer, Qwen3_5TextEngine,
    TextMlxEngineConstructors, ValidatedQwen3_5Artifact,
};
use runtime_integration::MlxMemoryLimits;

/// A 16-token prefill chunk keeps these short journeys fast while still
/// crossing the chunk boundary logic the production engine runs.
const SHORT_JOURNEY_PREFILL_CHUNK_SIZE_TOKENS: u32 = 16;

pub(super) fn construct_small_chunk_text_engine(
    validated_artifact: ValidatedQwen3_5Artifact,
    mlx_memory_limits: &MlxMemoryLimits,
    model_directory: &Path,
    image_pad_token_id: u32,
) -> Qwen3_5TextEngine {
    Qwen3_5TextEngine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.mlx_buffer_cache_limit_bytes(),
        None,
        PromptProcessingChunkSizer::for_chunk_size_tokens(SHORT_JOURNEY_PREFILL_CHUNK_SIZE_TOKENS)
            .expect("the test streaming prefill chunk size should be valid"),
        image_pad_token_id,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
    )
    .expect("the engine settings should be valid")
}

/// Drains up to `maximum_token_count` tokens and returns how many arrived,
/// requiring each id to sit inside the artifact's vocabulary. Exact token
/// values are deliberately not asserted so the journeys do not couple to one
/// quantization artifact.
pub(super) async fn count_tokens_within_vocabulary(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    request_id: RequestId,
    maximum_token_count: usize,
    model_vocabulary_size: u32,
    token_label: &str,
) -> usize {
    let mut generated_token_count = 0usize;
    while generated_token_count < maximum_token_count {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the request")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                assert!(
                    token_id < model_vocabulary_size,
                    "{token_label} token id {token_id} must be within vocabulary size {model_vocabulary_size}"
                );
                generated_token_count += 1;
            }
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::GenerationPreparationStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    generated_token_count
}
