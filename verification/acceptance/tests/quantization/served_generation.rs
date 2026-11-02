//! Serving a converted artifact through Astronomical's own engine: the
//! daemon's validator, a resident load, and a greedy Romeo and Juliet
//! continuation. Shared by every journey that ends in an artifact.

use std::path::Path;

use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration, Qwen3_5ResidentModel,
    Qwen3_5Tokenizer,
};

use super::support::{ROMEO_AND_JULIET_SOURCE, journey_runtime};

/// About 500 tokens of prompt, one prefill chunk.
const PROMPT_WORD_COUNT: usize = 360;
const DECODED_TOKEN_COUNT: usize = 50;
const MAXIMUM_OUTPUT_TOKENS: u32 = 4_096;

/// The production standard worker chunking tuple the daemon resolves for
/// the resident engine.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_PREFILL_LAYER_INTERVAL: u32 = 1;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;

/// Validates the artifact through the daemon's own validator, loads it as a
/// resident model, forwards a Romeo and Juliet prompt, and decodes greedily.
pub(crate) fn generate_greedy_continuation(artifact_directory: &Path) -> String {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(artifact_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the converted artifact should pass production validation");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the tokenizer should load from the converted artifact");
    let prompt_token_ids = tokenizer
        .encode_prompt(&romeo_and_juliet_prompt())
        .expect("the prompt should encode");
    let chunking = Qwen3_5ModelChunkingConfiguration::new(
        FULL_ATTENTION_KV_GROWTH_TOKENS,
        PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
        SSD_PAGING_PREFILL_LAYER_INTERVAL,
        SSD_PAGING_GENERATION_LAYER_INTERVAL,
    )
    .expect("the production chunking tuple is valid");
    let mut model = Qwen3_5ResidentModel::load(
        journey_runtime(),
        validated_artifact,
        artifact_directory,
        chunking,
    )
    .expect("the converted artifact should load as a resident model");
    model
        .materialize_complete_weights_matching_production()
        .expect("the converted weights should materialize");
    let generated_token_ids = greedy_decode(&model, &prompt_token_ids);
    decode_text(artifact_directory, &generated_token_ids)
}

/// Forwards the prompt once, then one token at a time, taking the argmax.
fn greedy_decode(model: &Qwen3_5ResidentModel, prompt_token_ids: &[u32]) -> Vec<u32> {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(
            i32::try_from(FULL_ATTENTION_KV_GROWTH_TOKENS).expect("fits"),
        )
        .expect("the decoder state should allocate");
    let mut generated_token_ids = Vec::with_capacity(DECODED_TOKEN_COUNT);
    let mut next_input = prompt_token_ids.to_vec();
    let mut position = 0_u32;
    for _ in 0..DECODED_TOKEN_COUNT {
        let logits = model
            .forward_chunk(&next_input, position, &mut decoder_state)
            .expect("the forward should complete");
        let argmax = model
            .runtime()
            .argmax_axis(&logits, -1)
            .expect("argmax")
            .to_vec_u32()
            .expect("read back");
        let next_token_id = *argmax.last().expect("at least one position");
        position += u32::try_from(next_input.len()).expect("fits");
        generated_token_ids.push(next_token_id);
        next_input = vec![next_token_id];
    }
    generated_token_ids
}

fn decode_text(artifact_directory: &Path, token_ids: &[u32]) -> String {
    tokenizers::Tokenizer::from_file(artifact_directory.join("tokenizer.json"))
        .expect("the copied tokenizer should load")
        .decode(token_ids, true)
        .expect("the continuation should decode")
}

/// The opening of the play in the chat shape the daemon serves.
pub(crate) fn romeo_and_juliet_prompt() -> String {
    let mut prompt_words = Vec::with_capacity(PROMPT_WORD_COUNT);
    for word in ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(PROMPT_WORD_COUNT)
    {
        prompt_words.push(word);
    }
    format!(
        "<|im_start|>user\nContinue this passage from Romeo and Juliet:\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt_words.join(" ")
    )
}
