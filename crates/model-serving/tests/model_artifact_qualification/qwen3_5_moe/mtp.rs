use std::collections::VecDeque;
use std::time::{Duration, Instant};

use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5Config, Qwen3_5ExecutionError, Qwen3_5Model,
    Qwen3_5MtpRequestState, RequestDecoderStateStack,
};
use astronomical_runtime_integration::MlxRuntime;

use crate::model_artifact_qualification::qwen3_5::mtp_support::run_one_layer_mtp_head_forward_qualification;

mod benchmark;
mod engine_support;
mod lifecycle;

use engine_support::{generate_with_mtp_engine, performance_counter_amount};

const IMAGE_PAD_TOKEN_ID: u32 = 248_056;
const WARMUP_OUTPUT_TOKEN_COUNT: usize = 2;
const MEASUREMENT_OUTPUT_TOKEN_COUNT: usize = 8;

struct GreedyDecodeMeasurement {
    elapsed_seconds: f64,
    generated_token_ids: Vec<u32>,
    verified_mtp_draft_count: usize,
    accepted_mtp_draft_count: usize,
}

impl GreedyDecodeMeasurement {
    fn tokens_per_second(&self) -> f64 {
        self.generated_token_ids.len() as f64 / self.elapsed_seconds.max(f64::EPSILON)
    }

    fn acceptance_rate(&self) -> f64 {
        if self.verified_mtp_draft_count == 0 {
            return 0.0;
        }
        self.accepted_mtp_draft_count as f64 / self.verified_mtp_draft_count as f64
    }
}

#[tokio::test]
#[ignore = "loads the complete local Qwen3.6-35B-A3B-oQ4e-mtp artifact and evaluates its MTP head"]
async fn should_evaluate_the_oq4e_mtp_head_from_target_pre_normalization_hidden_states() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_one_layer_mtp_head_forward_qualification(
            super::qwen3_6_35b_a3b_oq4e_mtp_model_directory(),
            "oq4e-moe-mtp-head",
        ),
    )
    .await
    .expect("the oQ4e MTP head qualification should finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads the local Qwen3.6-35B-A3B-oQ4e-mtp artifact and measures depth-one greedy MTP verification"]
async fn should_measure_depth_one_greedy_mtp_verification_cost() {
    tokio::time::timeout(
        Duration::from_secs(120),
        run_depth_one_greedy_mtp_verification_measurement(),
    )
    .await
    .expect("the depth-one MTP verification measurement should finish within 120 seconds");
}

async fn run_depth_one_greedy_mtp_verification_measurement() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let started_at = Instant::now();
    eprintln!("[oq4e-mtp-measure] status=start phase=artifact_validation");
    let model_directory = super::qwen3_6_35b_a3b_oq4e_mtp_model_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the local oQ4e MTP artifact should validate before native loading");
    let qwen3_5_config = validated_artifact.config().clone();
    eprintln!(
        "[oq4e-mtp-measure] status=progress phase=artifact_validated shards={} mtp_tensors={}",
        validated_artifact.shard_count(),
        validated_artifact.shard_index().mtp_tensor_count(),
    );

    eprintln!("[oq4e-mtp-measure] status=progress phase=runtime_init");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize for the oQ4e MTP measurement");
    eprintln!("[oq4e-mtp-measure] status=progress phase=model_load");
    let qwen3_5_model = Qwen3_5Model::load(runtime, validated_artifact, &model_directory, true)
        .expect("the complete local oQ4e MTP model should bind from validated descriptors");

    eprintln!("[oq4e-mtp-measure] status=progress phase=warmup");
    let _target_only_warmup = measure_target_only_greedy_decode(
        &qwen3_5_model,
        &qwen3_5_config,
        WARMUP_OUTPUT_TOKEN_COUNT,
    )
    .expect("target-only greedy warmup should complete successfully");
    let _mtp_verification_warmup = measure_depth_one_mtp_verified_greedy_decode(
        &qwen3_5_model,
        &qwen3_5_config,
        WARMUP_OUTPUT_TOKEN_COUNT,
    )
    .expect("depth-one MTP verification warmup should complete successfully");

    eprintln!("[oq4e-mtp-measure] status=progress phase=target_only_measurement_before_mtp");
    let target_only_measurement = measure_target_only_greedy_decode(
        &qwen3_5_model,
        &qwen3_5_config,
        MEASUREMENT_OUTPUT_TOKEN_COUNT,
    )
    .expect("target-only greedy decode should measure successfully");
    eprintln!(
        "[oq4e-mtp-measure] status=progress phase=mtp_verification_measurement target_only_before_mtp_tok_per_second={:.2}",
        target_only_measurement.tokens_per_second(),
    );
    let mtp_verification_measurement = measure_depth_one_mtp_verified_greedy_decode(
        &qwen3_5_model,
        &qwen3_5_config,
        MEASUREMENT_OUTPUT_TOKEN_COUNT,
    )
    .expect("depth-one MTP verification should measure successfully");
    eprintln!(
        "[oq4e-mtp-measure] status=progress phase=target_only_measurement_after_mtp mtp_verified_tok_per_second={:.2}",
        mtp_verification_measurement.tokens_per_second(),
    );
    let target_only_recheck_measurement = measure_target_only_greedy_decode(
        &qwen3_5_model,
        &qwen3_5_config,
        MEASUREMENT_OUTPUT_TOKEN_COUNT,
    )
    .expect("target-only greedy recheck should measure successfully");

    assert_eq!(
        mtp_verification_measurement.generated_token_ids,
        target_only_measurement.generated_token_ids,
        "MTP verification must preserve the target greedy token sequence"
    );
    assert_eq!(
        target_only_recheck_measurement.generated_token_ids,
        target_only_measurement.generated_token_ids,
        "target-only recheck must preserve the target greedy token sequence"
    );
    eprintln!(
        "[oq4e-mtp-measure] status=success elapsed_seconds={:.2} output_tokens={} target_only_before_mtp_tok_per_second={:.2} mtp_verified_tok_per_second={:.2} target_only_after_mtp_tok_per_second={:.2} mtp_verified_drafts={} mtp_accepted_drafts={} mtp_acceptance_rate={:.3}",
        started_at.elapsed().as_secs_f64(),
        MEASUREMENT_OUTPUT_TOKEN_COUNT,
        target_only_measurement.tokens_per_second(),
        mtp_verification_measurement.tokens_per_second(),
        target_only_recheck_measurement.tokens_per_second(),
        mtp_verification_measurement.verified_mtp_draft_count,
        mtp_verification_measurement.accepted_mtp_draft_count,
        mtp_verification_measurement.acceptance_rate(),
    );
    drop(qwen3_5_model);

    eprintln!("[oq4e-mtp-measure] status=progress phase=engine_mtp_smoke");
    let (engine_generated_token_ids, engine_performance_attribution) = generate_with_mtp_engine(
        &model_directory,
        MEASUREMENT_OUTPUT_TOKEN_COUNT as u16,
        true,
    )
    .await;
    assert_eq!(
        engine_generated_token_ids, target_only_measurement.generated_token_ids,
        "engine MTP prefix acceptance must preserve the target greedy token sequence"
    );
    let mtp_admitted_attempt_count = performance_counter_amount(
        &engine_performance_attribution,
        "mtp_admitted_attempt_count",
    );
    let mtp_accepted_draft_count =
        performance_counter_amount(&engine_performance_attribution, "mtp_accepted_draft_count");
    let mtp_rejected_draft_count =
        performance_counter_amount(&engine_performance_attribution, "mtp_rejected_draft_count");
    let mtp_operational_fallback_count = performance_counter_amount(
        &engine_performance_attribution,
        "mtp_operational_fallback_count",
    );
    assert_eq!(mtp_rejected_draft_count, 1);
    assert_eq!(mtp_operational_fallback_count, 0);
    assert_eq!(
        mtp_accepted_draft_count + mtp_rejected_draft_count + mtp_operational_fallback_count,
        mtp_admitted_attempt_count,
    );
}

fn measure_target_only_greedy_decode(
    qwen3_5_model: &Qwen3_5Model,
    qwen3_5_config: &Qwen3_5Config,
    output_token_count: usize,
) -> Result<GreedyDecodeMeasurement, Qwen3_5ExecutionError> {
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_config(qwen3_5_config);
    let mut next_input_token_id =
        prefill_measurement_prompt(qwen3_5_model, &mut request_decoder_state)?;
    let mut next_position_tokens =
        u32::try_from(super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.len() - 1)
            .expect("the measurement prompt prefix length should fit u32");
    let started_at = Instant::now();
    let mut generated_token_ids = Vec::with_capacity(output_token_count);
    for _ in 0..output_token_count {
        let target_logits = qwen3_5_model.forward_chunk(
            &[next_input_token_id],
            next_position_tokens,
            &mut request_decoder_state,
        )?;
        next_position_tokens = next_position_tokens
            .checked_add(1)
            .expect("the measurement position should not overflow");
        next_input_token_id = qwen3_5_model.greedy_token_id(&target_logits)?;
        generated_token_ids.push(next_input_token_id);
    }
    Ok(GreedyDecodeMeasurement {
        elapsed_seconds: started_at.elapsed().as_secs_f64(),
        generated_token_ids,
        verified_mtp_draft_count: 0,
        accepted_mtp_draft_count: 0,
    })
}

fn measure_depth_one_mtp_verified_greedy_decode(
    qwen3_5_model: &Qwen3_5Model,
    qwen3_5_config: &Qwen3_5Config,
    output_token_count: usize,
) -> Result<GreedyDecodeMeasurement, Qwen3_5ExecutionError> {
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_config(qwen3_5_config);
    let mut mtp_request_state = Qwen3_5MtpRequestState::empty();
    let final_prompt_token_id =
        prefill_measurement_prompt(qwen3_5_model, &mut request_decoder_state)?;
    let mut next_position_tokens =
        u32::try_from(super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.len() - 1)
            .expect("the measurement prompt prefix length should fit u32");

    let first_target_forward_output = qwen3_5_model
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &[final_prompt_token_id],
            next_position_tokens,
            &mut request_decoder_state,
        )?;
    next_position_tokens = next_position_tokens
        .checked_add(1)
        .expect("the measurement position should not overflow");
    let mut next_unforwarded_token_id =
        qwen3_5_model.greedy_token_id(first_target_forward_output.final_logits())?;
    let mut mtp_target_hidden_states = first_target_forward_output
        .pre_final_normalization_hidden_states()
        .retain()?;

    let started_at = Instant::now();
    let mut generated_token_ids = Vec::with_capacity(output_token_count);
    let mut verified_mtp_generated_token_ids = VecDeque::new();
    let mut verified_mtp_draft_count = 0_usize;
    let mut accepted_mtp_draft_count = 0_usize;
    while generated_token_ids.len() < output_token_count {
        if let Some(verified_mtp_generated_token_id) = verified_mtp_generated_token_ids.pop_front()
        {
            generated_token_ids.push(verified_mtp_generated_token_id);
            continue;
        }

        let current_generated_token_id = next_unforwarded_token_id;
        generated_token_ids.push(current_generated_token_id);
        let current_generated_token = qwen3_5_model
            .runtime()
            .array_from_u32(&[current_generated_token_id], &[1, 1])?;
        let mtp_forward_output = qwen3_5_model.forward_mtp_draft(
            &mtp_target_hidden_states,
            &current_generated_token,
            &mut mtp_request_state,
        )?;
        let draft_token_id = qwen3_5_model.greedy_token_id(mtp_forward_output.draft_logits())?;

        let target_state_checkpoint = request_decoder_state.checkpoint()?;
        let target_verify_start_position_tokens = next_position_tokens;
        let target_forward_output = qwen3_5_model
            .forward_chunk_with_all_position_logits_and_pre_final_normalization_hidden_states(
                &[current_generated_token_id, draft_token_id],
                next_position_tokens,
                &mut request_decoder_state,
            )?;
        next_position_tokens = next_position_tokens
            .checked_add(2)
            .expect("the measurement position should not overflow");
        let target_verify_token_ids = qwen3_5_model.greedy_token_ids(
            target_forward_output
                .all_position_logits()
                .expect("the MTP verification target forward should retain all logits"),
        )?;
        verified_mtp_draft_count += 1;

        if target_verify_token_ids[0] == draft_token_id {
            accepted_mtp_draft_count += 1;
            verified_mtp_generated_token_ids.push_back(draft_token_id);
            next_unforwarded_token_id = target_verify_token_ids[1];
            mtp_target_hidden_states = target_forward_output
                .pre_final_normalization_hidden_state_at(qwen3_5_model.runtime(), 1)?;
        } else {
            request_decoder_state.restore_checkpoint(target_state_checkpoint)?;
            next_position_tokens = target_verify_start_position_tokens;
            let replayed_current_target_forward_output = qwen3_5_model
                .forward_chunk_with_pre_final_normalization_hidden_states(
                    &[current_generated_token_id],
                    next_position_tokens,
                    &mut request_decoder_state,
                )?;
            next_position_tokens = next_position_tokens
                .checked_add(1)
                .expect("the measurement position should not overflow");
            next_unforwarded_token_id = qwen3_5_model
                .greedy_token_id(replayed_current_target_forward_output.final_logits())?;
            mtp_target_hidden_states = replayed_current_target_forward_output
                .pre_final_normalization_hidden_states()
                .retain()?;
        }
    }
    Ok(GreedyDecodeMeasurement {
        elapsed_seconds: started_at.elapsed().as_secs_f64(),
        generated_token_ids,
        verified_mtp_draft_count,
        accepted_mtp_draft_count,
    })
}

fn prefill_measurement_prompt(
    qwen3_5_model: &Qwen3_5Model,
    request_decoder_state: &mut RequestDecoderStateStack,
) -> Result<u32, Qwen3_5ExecutionError> {
    let final_prompt_token_position = super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.len() - 1;
    qwen3_5_model.prefill_chunck(
        &super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS[..final_prompt_token_position],
        0,
        request_decoder_state,
    )?;
    Ok(super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS[final_prompt_token_position])
}
