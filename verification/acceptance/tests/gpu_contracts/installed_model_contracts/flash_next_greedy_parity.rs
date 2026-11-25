//! Qwen3.8-Flash-Next on the streaming engine must choose the same greedy
//! tokens as the reference implementation (issue #2533).
//!
//! The reference fixture was written by the mlx-vlm `qwen4_exp` language
//! model run one layer at a time, greedy, on Romeo and Juliet text. Two
//! prompts cover the two attention regimes of the full-attention layers: a
//! short one, where every token attends to every earlier token, and one past
//! the sparse-attention budget (more than 512 complete four-token blocks),
//! where the indexer chooses which blocks each token reads. A wrong lane mix,
//! norm, n-gram row or block choice changes the chosen tokens within a few
//! steps, so matching every reference token is the whole contract.

use std::time::Duration;

use model_serving::{Qwen3_5ArtifactValidator, Qwen3_5StreamingModel};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;
use tokio::time::timeout;

const REFERENCE_FIXTURE_JSON: &str =
    include_str!("../../fixtures/flash_next_greedy_reference.json");
/// The memory limit the model is served at on a 48 GB Mac.
const MLX_MEMORY_LIMIT_BYTES: usize = 34 * 1_073_741_824;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 2 * 1_073_741_824;
/// Prefill chunk: small enough that the long prompt crosses chunk edges
/// while the sparse-attention block keys carry over between chunks.
const PREFILL_CHUNK_TOKEN_COUNT: usize = 1024;
const PARITY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads Qwen3.8-Flash-Next and runs real greedy replies"]
#[file_serial]
async fn should_choose_the_reference_greedy_tokens_on_the_streaming_engine() {
    timeout(PARITY_BUDGET, assert_greedy_tokens_match_the_reference())
        .await
        .expect("the Flash-Next parity journey must finish within its budget");
}

async fn assert_greedy_tokens_match_the_reference() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_installed_model_directory_by_id(
        crate::common::lane_mixing_sparse_moe_streaming_model_id(),
    );
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 64)
        .expect("the Flash-Next artifact should validate");
    let memory_limits = MlxMemoryLimits::new(MLX_MEMORY_LIMIT_BYTES, MLX_BUFFER_CACHE_LIMIT_BYTES)
        .expect("the MLX memory limits should be valid");
    let streaming_model = Qwen3_5StreamingModel::load(
        MlxRuntime::initialize(memory_limits).expect("the runtime should initialize"),
        validated_artifact,
        &model_directory,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("Flash-Next should load on the streaming engine");
    streaming_model
        .materialize_target_weights()
        .expect("the model weights should materialize");
    let reference_fixture: serde_json::Value =
        serde_json::from_str(REFERENCE_FIXTURE_JSON).expect("the reference fixture is JSON");
    let reference_prompts = reference_fixture["prompts"]
        .as_array()
        .expect("the reference fixture lists its prompts");
    assert_eq!(
        reference_prompts.len(),
        2,
        "both attention regimes are covered"
    );
    for reference_prompt in reference_prompts {
        let prompt_name = reference_prompt["name"].as_str().unwrap_or("unnamed");
        let prompt_token_ids = token_ids(&reference_prompt["prompt_token_ids"]);
        let reference_token_ids = token_ids(&reference_prompt["reference_token_ids"]);
        let chosen_token_ids = greedy_reply(
            &streaming_model,
            &prompt_token_ids,
            reference_token_ids.len(),
        );
        eprintln!(
            "[flash-next-parity] prompt={prompt_name} prompt_tokens={} chosen={chosen_token_ids:?}",
            prompt_token_ids.len()
        );
        assert_eq!(
            chosen_token_ids, reference_token_ids,
            "prompt {prompt_name}: the streaming engine must choose the reference tokens"
        );
    }
}

fn token_ids(token_id_list: &serde_json::Value) -> Vec<u32> {
    token_id_list
        .as_array()
        .expect("a token id list")
        .iter()
        .map(|token_id| {
            token_id
                .as_u64()
                .and_then(|token_id| u32::try_from(token_id).ok())
                .expect("a token id fits u32")
        })
        .collect()
}

/// Prefills in chunks and then decodes `reply_token_count` tokens, each the
/// highest-scoring one.
fn greedy_reply(
    streaming_model: &Qwen3_5StreamingModel,
    prompt_token_ids: &[u32],
    reply_token_count: usize,
) -> Vec<u32> {
    let mut request_decoder_state =
        crate::common::standard_request_decoder_state(&streaming_model.config);
    let mut last_logits = None;
    for chunk_start in (0..prompt_token_ids.len()).step_by(PREFILL_CHUNK_TOKEN_COUNT) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_TOKEN_COUNT).min(prompt_token_ids.len());
        last_logits = Some(
            streaming_model
                .forward_chunk(
                    &prompt_token_ids[chunk_start..chunk_end],
                    chunk_start as u32,
                    &mut request_decoder_state,
                )
                .expect("the prompt chunk should forward"),
        );
    }
    let mut next_logits = last_logits.expect("the prompt has at least one token");
    let mut chosen_token_ids = Vec::with_capacity(reply_token_count);
    for reply_index in 0..reply_token_count {
        let chosen_token_id = highest_scoring_token(&next_logits);
        chosen_token_ids.push(chosen_token_id);
        next_logits = streaming_model
            .forward_chunk(
                &[chosen_token_id],
                (prompt_token_ids.len() + reply_index) as u32,
                &mut request_decoder_state,
            )
            .expect("the reply token should forward");
    }
    chosen_token_ids
}

fn highest_scoring_token(logits: &mlx_bridge::MlxArray) -> u32 {
    let token_scores = logits.to_vec_f32().expect("the logits should read back");
    let mut best_token_index = 0;
    for (token_index, token_score) in token_scores.iter().enumerate() {
        if *token_score > token_scores[best_token_index] {
            best_token_index = token_index;
        }
    }
    u32::try_from(best_token_index).expect("a vocabulary index fits u32")
}
