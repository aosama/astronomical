//! The shared Qwen 3.5 core loaded with a folded ternary pack, as serving
//! loads it, and its greedy decode for the journeys over the pack (#2595).

use std::path::Path;

use mlx_bridge::MlxArray;
use model_serving::{
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5TextModel, RequestDecoderStateStack,
    TextModelChunkingConfiguration,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::support::PREFILL_CHUNK_TOKENS;

const CORE_ACTIVE_MEMORY_LIMIT_BYTES: usize = 24 * 1024 * 1024 * 1024;
const CORE_MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAXIMUM_OUTPUT_TOKENS: u32 = 4_096;
/// The key-value slab grows by this many tokens at a time.
const KEY_VALUE_GROWTH_TOKENS: i32 = 256;

/// The core with every layer of the pack at `pack_directory` held in memory.
pub(crate) fn loaded_shared_core(pack_directory: &Path) -> Qwen3_5TextModel {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            CORE_ACTIVE_MEMORY_LIMIT_BYTES,
            CORE_MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(pack_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the shared core validates the folded pack");
    // A reply token is submitted every few layers as serving submits it: one
    // graph a token decodes about 10% slower (dense_reply_token_verdict).
    let chunking = TextModelChunkingConfiguration::new(
        PREFILL_CHUNK_TOKENS as u32,
        0,
        config::DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
    )
    .expect("the chunking is valid");
    let shared_core = Qwen3_5TextModel::load_with_performance_attribution(
        runtime,
        validated_artifact,
        pack_directory,
        false,
        chunking,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the shared core loads the folded pack");
    shared_core
        .materialize_and_keep_every_dense_layer()
        .expect("every layer of the folded pack is kept");
    shared_core
}

/// Greedy reply tokens after `prompt_token_ids`, at most `maximum_token_count`
/// and ending at the first of `stop_token_ids`, which is not returned.
pub(crate) fn shared_core_greedy_tokens(
    shared_core: &Qwen3_5TextModel,
    prompt_token_ids: &[u32],
    maximum_token_count: usize,
    stop_token_ids: &[u32],
) -> Vec<u32> {
    let mut decoder_state = empty_shared_core_state(shared_core);
    let mut position = 0_u32;
    let mut reply_token_ids = Vec::with_capacity(maximum_token_count);
    let mut next_token_id = None;
    for prompt_chunk in prompt_token_ids.chunks(PREFILL_CHUNK_TOKENS) {
        let logits = core_logits_after(shared_core, prompt_chunk, position, &mut decoder_state);
        position += prompt_chunk.len() as u32;
        next_token_id = Some(picked_on_the_gpu(shared_core, &logits));
    }
    while let Some(picked_token_id) = next_token_id {
        if reply_token_ids.len() == maximum_token_count || stop_token_ids.contains(&picked_token_id)
        {
            break;
        }
        reply_token_ids.push(picked_token_id);
        let logits = core_logits_after(
            shared_core,
            &[picked_token_id],
            position,
            &mut decoder_state,
        );
        position += 1;
        next_token_id = Some(picked_on_the_gpu(shared_core, &logits));
    }
    reply_token_ids
}

/// The core's float32 logits, on the GPU, for the next token after
/// `token_ids` fed from `first_position`. The head runs in the same graph as
/// the layers, as serving runs a token.
fn core_logits_after(
    shared_core: &Qwen3_5TextModel,
    token_ids: &[u32],
    first_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
) -> MlxArray {
    let logits = shared_core
        .forward_chunk(token_ids, first_position, decoder_state)
        .expect("the shared core runs the forward");
    shared_core
        .runtime()
        .astype(&logits, mlx_bridge::MlxDtype::Float32)
        .expect("the logits widen")
}

/// The greedy pick, made on the GPU; only the token comes back.
pub(super) fn picked_on_the_gpu(shared_core: &Qwen3_5TextModel, logits: &MlxArray) -> u32 {
    *shared_core
        .runtime()
        .argmax_axis(logits, -1)
        .and_then(|token_ids| token_ids.to_vec_u32())
        .expect("the pick is made")
        .last()
        .expect("the logits have a row")
}

/// An empty request state for the shared core's layout.
pub(super) fn empty_shared_core_state(shared_core: &Qwen3_5TextModel) -> RequestDecoderStateStack {
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &shared_core.astro_decoder_state_cache_layout(),
        KEY_VALUE_GROWTH_TOKENS,
    )
    .expect("the decoder state allocates")
}
