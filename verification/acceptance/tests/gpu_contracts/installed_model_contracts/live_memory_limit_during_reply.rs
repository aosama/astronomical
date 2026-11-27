//! A reply whose memory limit moves while it runs (issue #2681) chooses the
//! same greedy tokens as a reply whose limit never moved, settles under the
//! lowered limit, and keeps going after a lower it cannot honour is refused.
//!
//! The limit moves at the three points the worker can move it: between two
//! prompt chunks (raised), between two reply tokens (lowered back), and once
//! more between reply tokens (a lower far below what the reply holds, which
//! must be refused with the memory it needs). The engine is driven a step at a
//! time, as the worker drives it, so each change lands exactly where the
//! worker would land it.

use std::path::Path;

use ipc_protocol::RequestId;
use model_serving::{
    GeneratedToken, InferenceEngine, InferenceEngineError, PromptProcessingChunkSizer,
    Qwen3_5ArtifactValidator, Qwen3_5TextEngine, TextInferenceRequest, TextMlxEngineConstructors,
};
use serial_test::serial;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// About 700 tokens of Shakespeare, so the prompt takes several chunks and a
/// change can land between two of them.
const PROMPT_WORD_COUNT: usize = 500;
const PREFILL_CHUNK_TOKEN_COUNT: u32 = 256;
const REPLY_TOKEN_COUNT: usize = 32;
const RAISE_HEADROOM_BYTES: u64 = 4 * GIGABYTE_BYTES;
const LOWER_BACK_AFTER_TOKEN_COUNT: usize = 8;
const REFUSED_LOWER_AFTER_TOKEN_COUNT: usize = 16;
/// Far below what any served model holds, so the lower must be refused.
const REFUSED_CEILING_BYTES: u64 = GIGABYTE_BYTES;
const GIGABYTE_BYTES: u64 = 1_073_741_824;
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;

/// Where a reply starts: under the whole machine less the raise's headroom
/// (the model is held whole), or under half the artifact (it streams).
#[derive(Clone, Copy)]
enum StartingCeiling {
    HoldsTheWholeModel,
    HalfTheArtifact,
}

#[tokio::test]
#[ignore = "loads the resident sparse model and moves its limit while a reply runs"]
#[serial]
async fn should_keep_the_greedy_reply_when_the_limit_moves_during_it_on_a_sparse_model_held_whole()
{
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    assert_the_reply_survives_live_limit_changes(
        &model_directory,
        StartingCeiling::HoldsTheWholeModel,
    )
    .await;
}

#[tokio::test]
#[ignore = "loads the small sparse model under half its size and moves its limit while a reply streams"]
#[serial]
async fn should_keep_the_greedy_reply_when_the_limit_moves_during_it_on_a_streaming_sparse_model() {
    let model_directory = crate::common::configured_small_sparse_moe_streaming_model_directory();
    assert_the_reply_survives_live_limit_changes(
        &model_directory,
        StartingCeiling::HalfTheArtifact,
    )
    .await;
}

#[tokio::test]
#[ignore = "loads the streamed dense model under half its size and moves its limit while a reply streams"]
#[serial]
async fn should_keep_the_greedy_reply_when_the_limit_moves_during_it_on_a_dense_model() {
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    assert_the_reply_survives_live_limit_changes(
        &model_directory,
        StartingCeiling::HalfTheArtifact,
    )
    .await;
}

#[tokio::test]
#[ignore = "loads the dense model with its drafter and moves its limit while a drafted reply runs"]
#[serial]
async fn should_keep_the_greedy_reply_when_the_limit_moves_during_a_drafted_reply() {
    let model_directory = crate::common::configured_installed_model_directory_by_id(
        test_harness::fixtures::dense_drafting_model_id(),
    );
    assert_the_reply_survives_live_limit_changes(
        &model_directory,
        StartingCeiling::HoldsTheWholeModel,
    )
    .await;
}

async fn assert_the_reply_survives_live_limit_changes(
    model_directory: &Path,
    starting_ceiling: StartingCeiling,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let starting_ceiling_bytes = starting_ceiling_bytes(model_directory, starting_ceiling).await;
    let prompt_token_ids = shakespeare_prompt_token_ids(model_directory);

    let mut unchanged_engine = load_engine(model_directory, starting_ceiling_bytes).await;
    let unchanged_reply = greedy_reply(&mut unchanged_engine, &prompt_token_ids, None).await;
    drop(unchanged_engine);

    let mut changed_engine = load_engine(model_directory, starting_ceiling_bytes).await;
    let changed_reply = greedy_reply(
        &mut changed_engine,
        &prompt_token_ids,
        Some(starting_ceiling_bytes),
    )
    .await;

    assert_eq!(
        changed_reply.reply_token_ids, unchanged_reply.reply_token_ids,
        "moving the limit during the reply must not change its greedy tokens"
    );
    assert!(changed_reply.was_raised_between_prompt_chunks);
    assert!(
        changed_reply.active_bytes_after_lowering_back <= starting_ceiling_bytes,
        "the reply must settle under the lowered limit: {} of {starting_ceiling_bytes} bytes active",
        changed_reply.active_bytes_after_lowering_back
    );
    eprintln!(
        "live-limit-reply model={} starting_ceiling_gb={:.2} refused_with={:?}",
        model_directory.display(),
        starting_ceiling_bytes as f64 / GIGABYTE_BYTES as f64,
        changed_reply.refusal_reason
    );
}

/// The reply's tokens and what the moving limit did along the way.
#[derive(Default)]
struct GreedyReply {
    reply_token_ids: Vec<u32>,
    was_raised_between_prompt_chunks: bool,
    active_bytes_after_lowering_back: u64,
    refusal_reason: Option<String>,
}

/// Decodes the reply a step at a time. With `moving_from_ceiling_bytes`, the
/// limit is raised at the first prompt chunk, lowered back to that ceiling
/// after a few tokens, and then asked to drop far below what the reply holds.
async fn greedy_reply(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    prompt_token_ids: &[u32],
    moving_from_ceiling_bytes: Option<u64>,
) -> GreedyReply {
    let request_id = RequestId::new(2_681);
    qwen3_5_engine
        .start_generation(greedy_request(request_id, prompt_token_ids))
        .await
        .expect("the reply should start");
    let mut greedy_reply = GreedyReply::default();
    while greedy_reply.reply_token_ids.len() < REPLY_TOKEN_COUNT {
        let generated_token = qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .unwrap_or_else(|decode_error| {
                panic!(
                    "the reply should advance (limit moving: {}, after {} tokens): {decode_error:?}",
                    moving_from_ceiling_bytes.is_some(),
                    greedy_reply.reply_token_ids.len()
                )
            });
        let Some(starting_ceiling_bytes) = moving_from_ceiling_bytes else {
            record_token(&mut greedy_reply, generated_token);
            continue;
        };
        let is_prefill_progress = matches!(generated_token, GeneratedToken::PrefillProgress { .. });
        record_token(&mut greedy_reply, generated_token);
        move_the_limit_at_this_step(
            qwen3_5_engine,
            &mut greedy_reply,
            is_prefill_progress,
            starting_ceiling_bytes,
        )
        .await;
    }
    greedy_reply
}

fn record_token(greedy_reply: &mut GreedyReply, generated_token: GeneratedToken) {
    if let GeneratedToken::TokenId { token_id, .. } = generated_token {
        greedy_reply.reply_token_ids.push(token_id);
    }
}

async fn move_the_limit_at_this_step(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    greedy_reply: &mut GreedyReply,
    is_prefill_progress: bool,
    starting_ceiling_bytes: u64,
) {
    let reply_token_count = greedy_reply.reply_token_ids.len();
    if is_prefill_progress && !greedy_reply.was_raised_between_prompt_chunks {
        qwen3_5_engine
            .update_mlx_memory_limit(starting_ceiling_bytes + RAISE_HEADROOM_BYTES)
            .await
            .expect("a raise between prompt chunks should be applied");
        greedy_reply.was_raised_between_prompt_chunks = true;
    } else if !is_prefill_progress && reply_token_count == LOWER_BACK_AFTER_TOKEN_COUNT {
        qwen3_5_engine
            .update_mlx_memory_limit(starting_ceiling_bytes)
            .await
            .expect("a lower back to the starting ceiling should be applied");
        greedy_reply.active_bytes_after_lowering_back = active_memory_bytes(qwen3_5_engine).await;
    } else if !is_prefill_progress && reply_token_count == REFUSED_LOWER_AFTER_TOKEN_COUNT {
        greedy_reply.refusal_reason = Some(refused_lower_reason(qwen3_5_engine).await);
    }
}

async fn refused_lower_reason(qwen3_5_engine: &mut Qwen3_5TextEngine) -> String {
    match qwen3_5_engine
        .update_mlx_memory_limit(REFUSED_CEILING_BYTES)
        .await
    {
        Err(InferenceEngineError::MlxMemoryLimitRejected {
            minimum_mlx_memory_ceiling_bytes,
            reason,
            ..
        }) => {
            assert!(minimum_mlx_memory_ceiling_bytes > REFUSED_CEILING_BYTES);
            assert!(
                reason.contains("needs at least") && reason.contains("while this request runs"),
                "the refusal must name the memory the reply needs: {reason}"
            );
            reason
        }
        other_outcome => panic!(
            "a lower far below what the reply holds must be refused cleanly, got {:?}",
            other_outcome.map(|_| "applied")
        ),
    }
}

async fn active_memory_bytes(qwen3_5_engine: &Qwen3_5TextEngine) -> u64 {
    qwen3_5_engine
        .collect_mlx_memory_telemetry()
        .await
        .expect("the engine should read its memory")
        .expect("a loaded engine reports its memory")
        .active_memory_bytes
}

fn greedy_request(request_id: RequestId, prompt_token_ids: &[u32]) -> TextInferenceRequest {
    TextInferenceRequest::new_sampling(
        request_id,
        prompt_token_ids.to_vec(),
        u16::try_from(REPLY_TOKEN_COUNT).expect("the reply length should fit the request"),
        0,
        1_000,
        None,
    )
    .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
    .with_thinking_configuration(false, None, Vec::new(), Vec::new())
}

/// The engine under `ceiling_bytes`, with no `astro_ssd_prompt_cache`, so each
/// reply reads its whole prompt.
async fn load_engine(model_directory: &Path, ceiling_bytes: u64) -> Qwen3_5TextEngine {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the model artifact should validate");
    let ceiling_bytes = usize::try_from(ceiling_bytes).expect("the ceiling should fit usize");
    let mut qwen3_5_engine = Qwen3_5TextEngine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        ceiling_bytes,
        ceiling_bytes,
        None,
        PromptProcessingChunkSizer::for_chunk_size_tokens(PREFILL_CHUNK_TOKEN_COUNT)
            .expect("the prompt chunk size should be valid"),
        IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
    )
    .expect("the engine should accept its configuration");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should load the model");
    qwen3_5_engine
}

async fn starting_ceiling_bytes(model_directory: &Path, starting_ceiling: StartingCeiling) -> u64 {
    match starting_ceiling {
        StartingCeiling::HoldsTheWholeModel => {
            let machine_ceiling_bytes =
                crate::common::sample_machine_mlx_memory_ceiling_bytes().await as u64;
            machine_ceiling_bytes - RAISE_HEADROOM_BYTES
        }
        StartingCeiling::HalfTheArtifact => artifact_weight_bytes(model_directory) / 2,
    }
}

fn artifact_weight_bytes(model_directory: &Path) -> u64 {
    std::fs::read_dir(model_directory)
        .expect("the model directory should be readable")
        .filter_map(Result::ok)
        .filter(|directory_entry| {
            directory_entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .filter_map(|directory_entry| directory_entry.metadata().ok())
        .map(|file_metadata| file_metadata.len())
        .sum()
}

fn shakespeare_prompt_token_ids(model_directory: &Path) -> Vec<u32> {
    tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the tokenizer should load")
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec()
}
