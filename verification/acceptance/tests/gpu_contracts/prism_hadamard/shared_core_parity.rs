//! A folded ternary pack served by the shared Qwen 3.5 core against the
//! folded engine that serves it today (issue #2595): the same Romeo and
//! Juliet prompt decoded greedily on both must give the same tokens. The
//! folded engine runs first and is let go before the core loads, because one
//! process holds one MLX runtime. Each side's reply speed is printed beside
//! the verdict, since the core must also decode as fast before the folded
//! engine retires.

use std::path::Path;
use std::time::Instant;

use mlx_bridge::MlxArray;
use model_serving::{
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5TextModel, RequestDecoderStateStack,
    TextModelChunkingConfiguration,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::support::{
    PREFILL_CHUNK_TOKENS, WARMUP_INPUT_TOKENS, loaded_journey_model, pack_directory,
    pack_tokenizer, romeo_and_juliet_prompt_token_ids,
};

/// Reply tokens compared, enough for a divergence in any layer's rounding to
/// surface as a different pick.
const COMPARED_TOKEN_COUNT: usize = 96;
const CORE_ACTIVE_MEMORY_LIMIT_BYTES: usize = 24 * 1024 * 1024 * 1024;
const CORE_MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAXIMUM_OUTPUT_TOKENS: u32 = 4_096;
/// The key-value slab grows by this many tokens at a time.
const KEY_VALUE_GROWTH_TOKENS: i32 = 256;

/// One side's greedy reply and how fast it decoded.
struct GreedyReply {
    token_ids: Vec<u32>,
    decode_tokens_per_second: f64,
}

#[test]
#[serial]
#[ignore = "loads the real folded pack twice, once on each engine, on the graphics processor"]
fn should_decode_a_folded_pack_on_the_shared_core_as_the_folded_engine_does() {
    let tokenizer = pack_tokenizer();
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS);
    let folded_engine_reply = folded_engine_greedy_reply(&prompt_token_ids);
    let shared_core_reply = shared_core_greedy_reply(&pack_directory(), &prompt_token_ids);
    let first_divergence = folded_engine_reply
        .token_ids
        .iter()
        .zip(&shared_core_reply.token_ids)
        .position(|(folded_token, core_token)| folded_token != core_token);
    eprintln!(
        "[prism-shared-core-parity] prompt_tokens={} compared_tokens={COMPARED_TOKEN_COUNT} first_divergence={first_divergence:?} folded_engine_decode_tokens_per_second={:.2} shared_core_decode_tokens_per_second={:.2}",
        prompt_token_ids.len(),
        folded_engine_reply.decode_tokens_per_second,
        shared_core_reply.decode_tokens_per_second,
    );
    eprintln!(
        "[prism-shared-core-parity] folded_engine_text={:?}",
        tokenizer.decode(&folded_engine_reply.token_ids, false)
    );
    eprintln!(
        "[prism-shared-core-parity] shared_core_text={:?}",
        tokenizer.decode(&shared_core_reply.token_ids, false)
    );
    assert_eq!(
        shared_core_reply.token_ids, folded_engine_reply.token_ids,
        "the shared core must pick the folded engine's tokens"
    );
}

fn folded_engine_greedy_reply(prompt_token_ids: &[i32]) -> GreedyReply {
    let folded_engine = loaded_journey_model();
    let mut session = folded_engine
        .begin_greedy_session(
            prompt_token_ids,
            PREFILL_CHUNK_TOKENS as u32,
            COMPARED_TOKEN_COUNT,
            &[],
        )
        .expect("the folded engine reads the prompt");
    let decode_started_at = Instant::now();
    let mut token_ids = Vec::with_capacity(COMPARED_TOKEN_COUNT);
    while let Some(token_id) = session.next_token().expect("the folded engine decodes") {
        token_ids.push(token_id);
    }
    GreedyReply {
        decode_tokens_per_second: token_ids.len() as f64
            / decode_started_at.elapsed().as_secs_f64(),
        token_ids,
    }
}

fn shared_core_greedy_reply(pack_directory: &Path, prompt_token_ids: &[i32]) -> GreedyReply {
    let shared_core = loaded_shared_core(pack_directory);
    let mut decoder_state = empty_shared_core_state(&shared_core);
    let prompt_token_ids: Vec<u32> = prompt_token_ids
        .iter()
        .map(|token_id| *token_id as u32)
        .collect();
    let mut next_token_id =
        greedy_pick_after(&shared_core, &prompt_token_ids, 0, &mut decoder_state);
    let mut position = prompt_token_ids.len() as u32;
    let decode_started_at = Instant::now();
    let mut token_ids = vec![next_token_id];
    while token_ids.len() < COMPARED_TOKEN_COUNT {
        next_token_id =
            greedy_pick_after(&shared_core, &[next_token_id], position, &mut decoder_state);
        position += 1;
        token_ids.push(next_token_id);
    }
    GreedyReply {
        decode_tokens_per_second: (token_ids.len() - 1) as f64
            / decode_started_at.elapsed().as_secs_f64(),
        token_ids,
    }
}

pub(super) fn loaded_shared_core(pack_directory: &Path) -> Qwen3_5TextModel {
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
    let chunking = TextModelChunkingConfiguration::new(PREFILL_CHUNK_TOKENS as u32, 0, 0)
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

/// The core's greedy pick after `token_ids` fed from `first_position`.
fn greedy_pick_after(
    shared_core: &Qwen3_5TextModel,
    token_ids: &[u32],
    first_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
) -> u32 {
    let forward_output = shared_core
        .forward_chunk_with_pre_final_normalization_hidden_states(
            token_ids,
            first_position,
            decoder_state,
        )
        .expect("the shared core runs the forward");
    let final_rows = forward_output.pre_final_normalization_hidden_states();
    let logits = shared_core
        .normalized_vocabulary_logits_for_every_row(&last_row(shared_core, final_rows))
        .expect("the head scores the last row");
    host_argmax(shared_core, &logits)
}

fn last_row(shared_core: &Qwen3_5TextModel, rows: &MlxArray) -> MlxArray {
    let shape = rows.shape();
    shared_core
        .runtime()
        .slice(
            rows,
            &[0, shape[1] - 1, 0],
            &[1, shape[1], shape[2]],
            &[1, 1, 1],
        )
        .expect("the last row slices")
}

fn host_argmax(shared_core: &Qwen3_5TextModel, logits: &MlxArray) -> u32 {
    let runtime = shared_core.runtime();
    let float32_logits = runtime
        .astype(logits, mlx_bridge::MlxDtype::Float32)
        .expect("the logits widen");
    let logit_values = runtime
        .array_to_vec_f32(&float32_logits)
        .expect("the logits read back");
    let mut best_token_id = 0_usize;
    for (token_id, logit_value) in logit_values.iter().enumerate() {
        if *logit_value > logit_values[best_token_id] {
            best_token_id = token_id;
        }
    }
    best_token_id as u32
}

/// An empty request state for the shared core's layout.
pub(super) fn empty_shared_core_state(shared_core: &Qwen3_5TextModel) -> RequestDecoderStateStack {
    RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &shared_core.astro_decoder_state_cache_layout(),
        KEY_VALUE_GROWTH_TOKENS,
    )
    .expect("the decoder state allocates")
}
