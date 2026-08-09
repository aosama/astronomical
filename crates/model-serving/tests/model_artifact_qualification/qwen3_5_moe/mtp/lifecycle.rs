use std::path::Path;
use std::time::Duration;

use astronomical_ipc_protocol::{MtpRuntimeState, RequestId};
use astronomical_model_serving::{
    GeneratedToken, GenerationFinalization, InferenceEngine, PerformanceAttribution, Qwen3_5Engine,
    Qwen3_5InferenceRequest,
};

use super::engine_support::{
    configured_mtp_artifact_test_inputs, generation_report_for_request, load_mtp_test_engine,
    performance_counter_amount,
};

#[tokio::test]
#[ignore = "loads the local MTP artifact twice and qualifies injection plus terminal lifecycle boundaries"]
async fn should_preserve_active_mtp_across_injection_and_request_lifecycle_boundaries() {
    tokio::time::timeout(Duration::from_secs(115), run_mtp_lifecycle_qualification())
        .await
        .expect("the MTP lifecycle qualification should finish within 115 seconds");
}

async fn run_mtp_lifecycle_qualification() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = super::super::configured_depth_one_mtp_model_artifact_directory();
    let configured_mtp_artifact_test_inputs = configured_mtp_artifact_test_inputs(&model_directory);
    let image_pad_token_id = configured_mtp_artifact_test_inputs.image_pad_token_id;

    eprintln!("[mtp-lifecycle] status=start phase=target_only_injection_control");
    let target_only_injected_token_ids = run_target_only_injection_control(
        &model_directory,
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        &configured_mtp_artifact_test_inputs.injected_feedback_token_ids,
        image_pad_token_id,
    )
    .await;

    eprintln!("[mtp-lifecycle] status=progress phase=active_engine_load");
    let (mut mtp_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(&model_directory, true, false).await;
    let engine_load_result = mtp_engine
        .load()
        .await
        .expect("the MTP lifecycle engine should load");
    assert_eq!(
        engine_load_result.mtp_runtime_state(),
        MtpRuntimeState::Active
    );

    let injection_request_id = RequestId::new(37_001);
    let active_mtp_injected_token_ids = run_injection_sequence(
        &mut mtp_engine,
        injection_request_id,
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        &configured_mtp_artifact_test_inputs.injected_feedback_token_ids,
        image_pad_token_id,
        true,
    )
    .await;
    eprintln!(
        "[mtp-lifecycle] status=diagnostic phase=injection target_only_token_ids={target_only_injected_token_ids:?} mtp_token_ids={active_mtp_injected_token_ids:?}",
    );
    assert!(
        active_mtp_injected_token_ids.len() >= 2,
        "MTP reset and reseed must emit a target-authorized token after injected input",
    );
    assert!(
        active_mtp_injected_token_ids.len() <= 6,
        "MTP reset and reseed must remain within the requested continuation limit",
    );
    let first_injected_continuation_mismatch = active_mtp_injected_token_ids
        .iter()
        .zip(&target_only_injected_token_ids)
        .enumerate()
        .find(
            |(_generated_token_index, (mtp_token_id, target_only_token_id))| {
                mtp_token_id != target_only_token_id
            },
        );
    eprintln!(
        "[mtp-lifecycle] status=diagnostic phase=injection exact_greedy_match={} first_greedy_mismatch={first_injected_continuation_mismatch:?}",
        first_injected_continuation_mismatch.is_none(),
    );
    let injection_report =
        generation_report_for_request(&performance_attribution_log_path, injection_request_id);
    let admitted_attempt_count =
        performance_counter_amount(&injection_report, "mtp_admitted_attempt_count");
    let accepted_draft_count =
        performance_counter_amount(&injection_report, "mtp_accepted_draft_count");
    let rejected_draft_count =
        performance_counter_amount(&injection_report, "mtp_rejected_draft_count");
    let operational_fallback_count =
        performance_counter_amount(&injection_report, "mtp_operational_fallback_count");
    assert!(admitted_attempt_count >= 1);
    assert_eq!(
        performance_counter_amount(&injection_report, "mtp_feedback_history_reseed_count"),
        1,
        "MTP must reseed request-local feedback history after injection",
    );
    assert_eq!(operational_fallback_count, 0);
    assert_eq!(
        accepted_draft_count + rejected_draft_count + operational_fallback_count,
        admitted_attempt_count,
        "every admitted MTP proposal must retain one verifier outcome across injection",
    );

    eprintln!("[mtp-lifecycle] status=progress phase=maximum_output_current_token");
    let current_token_terminal = run_request_to_terminal(
        &mut mtp_engine,
        RequestId::new(37_002),
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        image_pad_token_id,
        1,
    )
    .await;
    assert_eq!(current_token_terminal.generated_token_ids.len(), 1);
    assert!(
        current_token_terminal
            .generation_finalization
            .has_reportable_state()
    );

    eprintln!("[mtp-lifecycle] status=progress phase=maximum_output_queued_draft");
    let queued_draft_terminal = run_request_to_terminal(
        &mut mtp_engine,
        RequestId::new(37_003),
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        image_pad_token_id,
        2,
    )
    .await;
    assert_eq!(queued_draft_terminal.generated_token_ids.len(), 2);
    assert!(
        queued_draft_terminal
            .generation_finalization
            .has_reportable_state()
    );

    eprintln!("[mtp-lifecycle] status=progress phase=cancellation_with_mtp_state");
    let cancellation_request_id = RequestId::new(37_004);
    start_greedy_request(
        &mut mtp_engine,
        cancellation_request_id,
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        image_pad_token_id,
        8,
        false,
    )
    .await;
    let (_first_generated_token_id, first_generation_finalization) =
        decode_next_generated_token(&mut mtp_engine, cancellation_request_id).await;
    assert!(first_generation_finalization.is_none());
    let cancellation_finalization = mtp_engine
        .cancel_generation(cancellation_request_id)
        .await
        .expect("active MTP cancellation should release the request");
    assert!(cancellation_finalization.has_reportable_state());

    eprintln!("[mtp-lifecycle] status=progress phase=eos_and_engine_reuse");
    let eos_terminal = run_request_to_terminal(
        &mut mtp_engine,
        RequestId::new(37_005),
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        image_pad_token_id,
        64,
    )
    .await;
    let final_eos_token_id = eos_terminal
        .generated_token_ids
        .last()
        .copied()
        .expect("the EOS request should emit at least one token");
    assert!(
        configured_mtp_artifact_test_inputs
            .end_of_sequence_token_ids
            .contains(&final_eos_token_id),
        "the request should terminate on a certified Qwen EOS token"
    );
    assert!(eos_terminal.generation_finalization.has_reportable_state());

    let reused_engine_terminal = run_request_to_terminal(
        &mut mtp_engine,
        RequestId::new(37_006),
        &configured_mtp_artifact_test_inputs.short_prompt_token_ids,
        image_pad_token_id,
        1,
    )
    .await;
    assert_eq!(reused_engine_terminal.generated_token_ids.len(), 1);
    assert!(
        reused_engine_terminal
            .generation_finalization
            .has_reportable_state()
    );
    eprintln!("[mtp-lifecycle] status=success");
}

async fn run_target_only_injection_control(
    model_directory: &Path,
    short_prompt_token_ids: &[u32],
    injected_feedback_token_ids: &[u32],
    image_pad_token_id: u32,
) -> Vec<u32> {
    let (mut target_only_engine, _temporary_log_directory, _performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, false, false).await;
    let engine_load_result = target_only_engine
        .load()
        .await
        .expect("the target-only injection control engine should load");
    assert_eq!(
        engine_load_result.mtp_runtime_state(),
        MtpRuntimeState::Disabled
    );
    run_injection_sequence(
        &mut target_only_engine,
        RequestId::new(37_000),
        short_prompt_token_ids,
        injected_feedback_token_ids,
        image_pad_token_id,
        false,
    )
    .await
}

async fn run_injection_sequence(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    short_prompt_token_ids: &[u32],
    injected_feedback_token_ids: &[u32],
    image_pad_token_id: u32,
    performance_attribution_enabled: bool,
) -> Vec<u32> {
    start_greedy_request(
        engine,
        request_id,
        short_prompt_token_ids,
        image_pad_token_id,
        6,
        performance_attribution_enabled,
    )
    .await;
    let (first_generated_token_id, first_generation_finalization) =
        decode_next_generated_token(engine, request_id).await;
    assert!(first_generation_finalization.is_none());
    engine
        .inject_input_tokens(request_id, injected_feedback_token_ids.to_vec())
        .await
        .expect("same-request feedback injection should succeed");

    let mut generated_token_ids = vec![first_generated_token_id];
    while generated_token_ids.len() < 6 {
        let (generated_token_id, generation_finalization) =
            decode_next_generated_token(engine, request_id).await;
        generated_token_ids.push(generated_token_id);
        if generation_finalization.is_some() {
            break;
        }
    }
    generated_token_ids
}

async fn run_request_to_terminal(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    short_prompt_token_ids: &[u32],
    image_pad_token_id: u32,
    maximum_output_tokens: u16,
) -> TerminalGeneration {
    start_greedy_request(
        engine,
        request_id,
        short_prompt_token_ids,
        image_pad_token_id,
        maximum_output_tokens,
        false,
    )
    .await;
    let mut generated_token_ids = Vec::with_capacity(maximum_output_tokens as usize);
    loop {
        let (generated_token_id, generation_finalization) =
            decode_next_generated_token(engine, request_id).await;
        generated_token_ids.push(generated_token_id);
        if let Some(generation_finalization) = generation_finalization {
            return TerminalGeneration {
                generated_token_ids,
                generation_finalization,
            };
        }
    }
}

async fn start_greedy_request(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    short_prompt_token_ids: &[u32],
    image_pad_token_id: u32,
    maximum_output_tokens: u16,
    performance_attribution_enabled: bool,
) {
    let inference_request = Qwen3_5InferenceRequest::new(
        request_id,
        short_prompt_token_ids.to_vec(),
        maximum_output_tokens,
    )
    .with_image_pad_token_id(image_pad_token_id);
    let inference_request = if performance_attribution_enabled {
        inference_request.with_performance_attribution(PerformanceAttribution::enabled())
    } else {
        inference_request
    };
    engine
        .start_generation(inference_request)
        .await
        .expect("the reused engine should accept the greedy request");
}

async fn decode_next_generated_token(
    engine: &mut Qwen3_5Engine,
    request_id: RequestId,
) -> (u32, Option<GenerationFinalization>) {
    loop {
        match engine
            .decode_next_token(request_id)
            .await
            .expect("the MTP lifecycle request should advance")
        {
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => return (token_id, generation_finalization),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::EndOfSequence => {
                panic!("the Qwen engine should attach terminal state to an emitted token")
            }
        }
    }
}

struct TerminalGeneration {
    generated_token_ids: Vec<u32>,
    generation_finalization: GenerationFinalization,
}
