use std::time::Duration;

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Model};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::timeout;

use super::compact_prefill::maximum_absolute_difference;
use super::prompt::prepare_reproduced_long_prompt_token_ids;

const ACCEPTANCE_PROMPT_TOKEN_COUNT: usize = 4_097;
const ACCEPTANCE_OUTPUT_TOKEN_COUNT: u16 = 512;
const ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "loads the configured model and requires exact final-prefill logits for 2048 and 4096 chunks"]
async fn should_preserve_exact_final_prefill_logits_between_fixed_prefill_sizes() {
    timeout(
        ACCEPTANCE_TIMEOUT,
        assert_exact_final_prefill_logit_parity(),
    )
    .await
    .expect("the final-prefill logit parity contract must finish within 115 seconds");
}

async fn assert_exact_final_prefill_logit_parity() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let configured_model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let prompt_token_ids = prepare_reproduced_long_prompt_token_ids(
        ACCEPTANCE_PROMPT_TOKEN_COUNT,
        ACCEPTANCE_OUTPUT_TOKEN_COUNT,
    );
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(
            &configured_model_directory,
            u32::from(ACCEPTANCE_OUTPUT_TOKEN_COUNT),
        )
        .expect("the Ornith artifact should validate before final-logit parity acceptance");
    let qwen3_5_config = validated_artifact.config().clone();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the machine-derived final-logit parity runtime should initialize");
    eprintln!("[prefill-logit-parity] status=progress phase=model_load");
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &configured_model_directory,
        false,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the configured model should load for final-logit parity acceptance");
    let (baseline_highest_logit_token_id, baseline_final_logits) =
        final_prefill_logits_for_chunk_size(
            &qwen3_5_model,
            &qwen3_5_config,
            &prompt_token_ids,
            2_048,
        );
    let (candidate_highest_logit_token_id, candidate_final_logits) =
        final_prefill_logits_for_chunk_size(
            &qwen3_5_model,
            &qwen3_5_config,
            &prompt_token_ids,
            4_096,
        );
    let maximum_absolute_logit_delta =
        maximum_absolute_difference(&baseline_final_logits, &candidate_final_logits);
    assert!(
        maximum_absolute_logit_delta.is_finite(),
        "full-chunk logit delta must remain finite"
    );
    assert_eq!(
        maximum_absolute_logit_delta, 0.0,
        "fixed 2048 and 4096 prefill sizes must produce exact final logits"
    );
    eprintln!(
        "[prefill-logit-parity] status=success baseline_highest_logit_token_id={baseline_highest_logit_token_id} candidate_highest_logit_token_id={candidate_highest_logit_token_id} maximum_absolute_logit_delta={maximum_absolute_logit_delta:.6}"
    );
}

fn final_prefill_logits_for_chunk_size(
    qwen3_5_model: &Qwen3_5Model,
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
    prompt_token_ids: &[u32],
    prefill_chunk_tokens: usize,
) -> (u32, Vec<f32>) {
    let mut request_decoder_state = crate::common::standard_request_decoder_state(qwen3_5_config);
    let final_prompt_token_index = prompt_token_ids
        .len()
        .checked_sub(1)
        .expect("the diagnostic prompt should contain a final decode-seeding token");
    for prefill_chunk_start in (0..final_prompt_token_index).step_by(prefill_chunk_tokens) {
        let prefill_chunk_end = prefill_chunk_start
            .saturating_add(prefill_chunk_tokens)
            .min(final_prompt_token_index);
        qwen3_5_model
            .prefill_chunk(
                &prompt_token_ids[prefill_chunk_start..prefill_chunk_end],
                u32::try_from(prefill_chunk_start)
                    .expect("the diagnostic prompt position should fit u32"),
                &mut request_decoder_state,
            )
            .expect("the diagnostic prefill chunk should complete");
    }
    let final_position_logits = qwen3_5_model
        .forward_chunk(
            &prompt_token_ids[final_prompt_token_index..],
            u32::try_from(final_prompt_token_index)
                .expect("the diagnostic final prompt position should fit u32"),
            &mut request_decoder_state,
        )
        .expect("the diagnostic final prompt token should produce logits");
    let highest_logit_token_id = qwen3_5_model
        .highest_logit_token_id(&final_position_logits)
        .expect("the diagnostic final logits should produce a highest-logit token");
    let final_logits = final_position_logits
        .to_vec_f32()
        .expect("the diagnostic final logits should materialize to CPU");
    assert!(
        final_logits.iter().all(|logit| logit.is_finite()),
        "the diagnostic final logits must remain finite"
    );
    (highest_logit_token_id, final_logits)
}
