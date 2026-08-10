use std::time::Instant;

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Model, Qwen3_5MtpRequestState};
use astronomical_runtime_integration::MlxRuntime;

use crate::model_artifact_qualification::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS;

pub(crate) async fn run_one_layer_mtp_head_forward_qualification(
    model_directory: std::path::PathBuf,
    progress_log_prefix: &str,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    eprintln!("[{progress_log_prefix}] status=start phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the local oQ4e MTP artifact should validate before native loading");
    let qwen3_5_config = validated_artifact.config().clone();
    eprintln!(
        "[{progress_log_prefix}] status=progress phase=artifact_validated shards={} mtp_tensors={}",
        validated_artifact.shard_count(),
        validated_artifact.shard_index().mtp_tensor_count(),
    );

    eprintln!("[{progress_log_prefix}] status=progress phase=runtime_init");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for the oQ4e MTP head");
    eprintln!("[{progress_log_prefix}] status=progress phase=model_load");
    let qwen3_5_model = Qwen3_5Model::load(
        runtime,
        validated_artifact,
        &model_directory,
        true,
        crate::common::standard_qwen3_5_model_chunking_configuration(),
    )
    .expect("the complete local oQ4e MTP model should bind from validated descriptors");

    let mut request_decoder_state = crate::common::standard_request_decoder_state(&qwen3_5_config);
    eprintln!("[{progress_log_prefix}] status=progress phase=target_forward");
    let first_prompt_target_forward_output = qwen3_5_model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &[SAY_HI_PROMPT_TOKEN_IDS[0]],
            0,
            &mut request_decoder_state,
        )
        .expect("the target forward should retain the pre-final-normalization hidden row");
    let shifted_prompt_token_indices = qwen3_5_model
        .runtime()
        .array_from_u32(&[SAY_HI_PROMPT_TOKEN_IDS[1]], &[1, 1])
        .expect("the shifted prompt token should fit the direct MLX index representation");
    let mut mtp_request_state = Qwen3_5MtpRequestState::empty_with_growth_tokens(256)
        .expect("the test MTP growth should be valid");
    qwen3_5_model
        .prefill_mtp_history(
            first_prompt_target_forward_output.pre_final_normalization_hidden_states(),
            &shifted_prompt_token_indices,
            &mut mtp_request_state,
        )
        .expect("the MTP head should commit shifted prompt history without draft logits");
    assert_eq!(mtp_request_state.committed_token_count(), 1);
    let target_forward_output = qwen3_5_model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &[SAY_HI_PROMPT_TOKEN_IDS[1]],
            1,
            &mut request_decoder_state,
        )
        .expect("the second prompt token should produce the live MTP seed hidden row");
    let next_token_id = qwen3_5_model
        .greedy_token_id(target_forward_output.final_logits())
        .expect("the target logits should produce one MTP seed token");
    let next_token_indices = qwen3_5_model
        .runtime()
        .array_from_u32(&[next_token_id], &[1, 1])
        .expect("the MTP seed token should fit the direct MLX index representation");
    eprintln!("[{progress_log_prefix}] status=progress phase=mtp_forward");
    let mtp_forward_output = qwen3_5_model
        .forward_mtp_draft(
            target_forward_output.pre_final_normalization_hidden_states(),
            &next_token_indices,
            &mut mtp_request_state,
        )
        .expect("the native MTP head should evaluate from the target hidden row");

    assert_eq!(
        mtp_forward_output.draft_logits().shape(),
        [1, 1, qwen3_5_config.vocabulary_size() as i32],
    );
    assert_eq!(
        mtp_forward_output
            .post_normalization_hidden_states()
            .shape(),
        [1, 1, qwen3_5_config.hidden_size() as i32],
    );
    assert_eq!(mtp_request_state.committed_token_count(), 2);
    mtp_request_state
        .reset_with_growth_tokens(256)
        .expect("injected-input reset should replace MTP history with empty state");
    assert_eq!(mtp_request_state.committed_token_count(), 0);
    eprintln!(
        "[{progress_log_prefix}] status=success elapsed_seconds={:.2}",
        started_at.elapsed().as_secs_f64()
    );
}
