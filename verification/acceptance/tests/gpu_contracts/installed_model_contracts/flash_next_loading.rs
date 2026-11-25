//! Loading Qwen3.8-Flash-Next on the streaming engine for its contracts:
//! the configured model at the 34 GiB limit it is served at on a 48 GB Mac.

use model_serving::{Qwen3_5ArtifactValidator, Qwen3_5StreamingModel};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

const MLX_MEMORY_LIMIT_BYTES: usize = 34 * 1_073_741_824;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 1_073_741_824;

/// The configured Flash-Next model's folder.
pub(super) fn flash_next_model_directory() -> std::path::PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::lane_mixing_sparse_moe_streaming_model_id(),
    )
}

/// Validates, loads and materializes the model.
pub(super) fn load_flash_next(model_directory: &std::path::Path) -> Qwen3_5StreamingModel {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 64)
        .expect("the Flash-Next artifact should validate");
    let memory_limits = MlxMemoryLimits::new(MLX_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
        .expect("the MLX memory limits should be valid");
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(memory_limits).expect("the runtime should initialize"),
        validated_artifact,
        model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("Flash-Next should load on the streaming engine");
    streaming_model
        .materialize_target_weights()
        .expect("the model weights should materialize");
    streaming_model
}

/// Forwards one prompt chunk starting at `chunk_start` and returns its last
/// logits.
pub(super) fn forward_prompt_chunk(
    streaming_model: &Qwen3_5StreamingModel,
    prompt_token_ids: &[u32],
    chunk_start: usize,
    chunk_token_count: usize,
    request_decoder_state: &mut model_serving::RequestDecoderStateStack,
) -> mlx_bridge::MlxArray {
    let chunk_end = (chunk_start + chunk_token_count).min(prompt_token_ids.len());
    streaming_model
        .forward_chunk(
            &prompt_token_ids[chunk_start..chunk_end],
            chunk_start as u32,
            request_decoder_state,
        )
        .expect("the prompt chunk should forward")
}
