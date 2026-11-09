//! Serving a dense model from the SSD must not change its answer (issue #1985).
//!
//! The streaming engine serves a dense model whose text weights do not fit
//! the limit: it reads embedding rows from the SSD instead of keeping the
//! table resident. The resident engine keeps everything in memory. For the
//! same Shakespeare prompt, both must produce bit-identical final logits.
//!
//! The streaming model's on-disk embeddings do not depend on the limit, so
//! both models load in this one process, one after the other.

use std::time::Duration;

use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ResidentModel, Qwen3_5StreamingModel,
};
use astronomical_runtime_integration::MlxRuntime;
use serial_test::serial;
use tokio::time::timeout;

#[path = "../../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

/// Words of the Shakespeare fixture: about 1,000 tokens, more than one
/// standard prefill chunk, so the comparison crosses a chunk boundary.
const PROMPT_WORD_COUNT: usize = 600;
const OUTPUT_TOKEN_COUNT: u32 = 64;
const PARITY_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "loads the streamed_dense model twice and requires exact final-logit parity"]
#[serial]
async fn should_give_the_resident_logits_when_dense_embeddings_are_read_from_disk() {
    timeout(PARITY_BUDGET, assert_dense_streaming_logit_parity())
        .await
        .expect("the dense streaming parity journey must finish within its budget");
}

async fn assert_dense_streaming_logit_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_streamed_dense_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;

    eprintln!("[dense-parity] status=progress step=1/2 engine=streaming");
    let (streaming_highest_token_id, streaming_final_logits) = {
        let (validated_artifact, qwen3_5_config) = validated_dense_artifact(&model_directory);
        let streaming_model = Qwen3_5StreamingModel::load(
            MlxRuntime::initialize(mlx_memory_limits)
                .expect("the streaming runtime should initialize"),
            validated_artifact,
            &model_directory,
            crate::common::standard_qwen3_5_model_chunking_configuration(),
        )
        .expect("the dense model should load on the streaming engine");
        final_logits_after_prompt(
            &qwen3_5_config,
            &prompt_token_ids,
            |token_ids, start, state| streaming_model.forward_chunk(token_ids, start, state),
        )
    };
    eprintln!("[dense-parity] status=progress step=2/2 engine=resident");
    let (resident_highest_token_id, resident_final_logits) = {
        let (validated_artifact, qwen3_5_config) = validated_dense_artifact(&model_directory);
        let resident_model = Qwen3_5ResidentModel::load(
            MlxRuntime::initialize(mlx_memory_limits)
                .expect("the resident runtime should initialize"),
            validated_artifact,
            &model_directory,
            crate::common::standard_qwen3_5_model_chunking_configuration(),
        )
        .expect("the dense model should load on the resident engine");
        final_logits_after_prompt(
            &qwen3_5_config,
            &prompt_token_ids,
            |token_ids, start, state| resident_model.forward_chunk(token_ids, start, state),
        )
    };

    assert_eq!(streaming_final_logits.len(), resident_final_logits.len());
    assert_eq!(streaming_highest_token_id, resident_highest_token_id);
    assert!(
        streaming_final_logits == resident_final_logits,
        "reading dense embeddings from disk must give bit-identical final logits"
    );
    eprintln!(
        "[dense-parity] status=success prompt_tokens={} highest_token_id={streaming_highest_token_id}",
        prompt_token_ids.len()
    );
}

fn validated_dense_artifact(
    model_directory: &std::path::Path,
) -> (
    astronomical_model_serving::ValidatedQwen3_5Artifact,
    Qwen3_5Config,
) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, OUTPUT_TOKEN_COUNT)
        .expect("the streamed dense artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    (validated_artifact, qwen3_5_config)
}

fn shakespeare_prompt_token_ids(model_directory: &std::path::Path) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the dense model's tokenizer should load");
    tokenizer
        .encode(romeo_and_juliet::summary_prompt(PROMPT_WORD_COUNT), false)
        .expect("the Shakespeare prompt should encode")
        .get_ids()
        .to_vec()
}

/// Runs the prompt one standard chunk at a time and returns the highest-logit
/// token and the final logits after its last token.
fn final_logits_after_prompt(
    qwen3_5_config: &Qwen3_5Config,
    prompt_token_ids: &[u32],
    forward_chunk: impl Fn(
        &[u32],
        u32,
        &mut astronomical_model_serving::RequestDecoderStateStack,
    ) -> Result<
        mlx_c_rust::MlxArray,
        astronomical_model_serving::Qwen3_5ExecutionError,
    >,
) -> (usize, Vec<f32>) {
    const PREFILL_CHUNK_TOKEN_COUNT: usize = 256;
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let mut final_logits = None;
    for chunk_start in (0..prompt_token_ids.len()).step_by(PREFILL_CHUNK_TOKEN_COUNT) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_TOKEN_COUNT).min(prompt_token_ids.len());
        final_logits = Some(
            forward_chunk(
                &prompt_token_ids[chunk_start..chunk_end],
                u32::try_from(chunk_start).expect("the chunk start should fit u32"),
                &mut request_decoder_state,
            )
            .expect("the prompt chunk should run"),
        );
    }
    let final_logits = final_logits
        .expect("the prompt should have at least one chunk")
        .to_vec_f32()
        .expect("the final logits should be readable");
    assert!(final_logits.iter().all(|logit| logit.is_finite()));
    let mut highest_token_id = 0;
    for (token_id, logit) in final_logits.iter().enumerate() {
        if *logit > final_logits[highest_token_id] {
            highest_token_id = token_id;
        }
    }
    (highest_token_id, final_logits)
}
