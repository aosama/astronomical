//! A folded ternary pack served by the shared Qwen 3.5 core against the
//! folded engine that serves it today (issue #2595): the same Romeo and
//! Juliet prompt decoded greedily on both must give the same tokens, except
//! where the model's top two choices are so close that float16 rounding
//! decides between them. The two engines round a few steps differently (the
//! folded engine fuses the DeltaNet rate projections into one matmul, for
//! one), so where the replies part, the core's logit for its own pick must be
//! within `PARTING_LOGIT_MARGIN` of its logit for the folded engine's pick. The
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
/// The most the core's logit for its own pick may exceed its logit for the
/// folded engine's pick where the two replies part: a near-tie that float16
/// rounding of a few stages settles, not a difference in what the model says.
const PARTING_LOGIT_MARGIN: f32 = 0.0625;
/// The key-value slab grows by this many tokens at a time.
const KEY_VALUE_GROWTH_TOKENS: i32 = 256;

/// One side's greedy reply and how fast it decoded.
struct GreedyReply {
    token_ids: Vec<u32>,
    decode_tokens_per_second: f64,
    /// Where the core first picked another token than the folded engine:
    /// the reply position and how far its own pick's logit led the folded
    /// engine's pick's.
    parting: Option<(usize, f32)>,
}

#[test]
#[serial]
#[ignore = "loads the real folded pack twice, once on each engine, on the graphics processor"]
fn should_decode_a_folded_pack_on_the_shared_core_as_the_folded_engine_does() {
    let tokenizer = pack_tokenizer();
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS);
    let folded_engine_reply = folded_engine_greedy_reply(&prompt_token_ids);
    let shared_core_reply = shared_core_greedy_reply(
        &pack_directory(),
        &prompt_token_ids,
        &folded_engine_reply.token_ids,
    );
    eprintln!(
        "[prism-shared-core-parity] prompt_tokens={} compared_tokens={COMPARED_TOKEN_COUNT} parting={:?} folded_engine_decode_tokens_per_second={:.2} shared_core_decode_tokens_per_second={:.2}",
        prompt_token_ids.len(),
        shared_core_reply.parting,
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
    if let Some((parting_position, parting_logit_margin)) = shared_core_reply.parting {
        assert!(
            parting_logit_margin <= PARTING_LOGIT_MARGIN,
            "the replies part at token {parting_position} by a logit margin of {parting_logit_margin}, more than rounding"
        );
    }
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
        parting: None,
    }
}

fn shared_core_greedy_reply(
    pack_directory: &Path,
    prompt_token_ids: &[i32],
    folded_engine_token_ids: &[u32],
) -> GreedyReply {
    let shared_core = loaded_shared_core(pack_directory);
    let mut decoder_state = empty_shared_core_state(&shared_core);
    let prompt_token_ids: Vec<u32> = prompt_token_ids
        .iter()
        .map(|token_id| *token_id as u32)
        .collect();
    let mut step_logits = core_logits_after(&shared_core, &prompt_token_ids, 0, &mut decoder_state);
    let mut position = prompt_token_ids.len() as u32;
    let decode_started_at = Instant::now();
    let mut token_ids = Vec::with_capacity(COMPARED_TOKEN_COUNT);
    let mut parting = None;
    loop {
        let next_token_id = argmax(&step_logits);
        let folded_engine_token_id = folded_engine_token_ids[token_ids.len()];
        if parting.is_none() && next_token_id != folded_engine_token_id {
            let logit_margin =
                step_logits[next_token_id as usize] - step_logits[folded_engine_token_id as usize];
            parting = Some((token_ids.len(), logit_margin));
        }
        token_ids.push(next_token_id);
        if token_ids.len() == COMPARED_TOKEN_COUNT {
            break;
        }
        step_logits =
            core_logits_after(&shared_core, &[next_token_id], position, &mut decoder_state);
        position += 1;
    }
    GreedyReply {
        decode_tokens_per_second: (token_ids.len() - 1) as f64
            / decode_started_at.elapsed().as_secs_f64(),
        token_ids,
        parting,
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

/// The core's float32 logits for the next token after `token_ids` fed from
/// `first_position`.
fn core_logits_after(
    shared_core: &Qwen3_5TextModel,
    token_ids: &[u32],
    first_position: u32,
    decoder_state: &mut RequestDecoderStateStack,
) -> Vec<f32> {
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
    let runtime = shared_core.runtime();
    let float32_logits = runtime
        .astype(&logits, mlx_bridge::MlxDtype::Float32)
        .expect("the logits widen");
    runtime
        .array_to_vec_f32(&float32_logits)
        .expect("the logits read back")
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

fn argmax(logit_values: &[f32]) -> u32 {
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
